import numpy as np 
import glob

import cv2
from PIL import Image
from scipy.ndimage.measurements import minimum
from skimage import io, transform
import scipy.ndimage as ndi

import h5py 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import morphology as mor

class Data_Handler(Dataset):

    def __init__(self):

        ####
        self.fns = []
        self.selection_weights = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #### random selection parameters
        self.selection_size = 5
        self.randomize = True
        self.random_selection = False
        self.scale_range = [.5,1.]
        self.outsize = [128,128]
        self.rand_flip = [0.5,0.5]
        ## signal Segmentation params
        self.threshold = None
        self.area_range = None
        self.fill_area = None
        self.imclose = None 
        self.force_label = False
        ## params for calculating loss 
        self.method = None
        self.train_channels = None
        ## Variables for buffering 
        self.use_in_memory = True
        self.in_memory = []
        ## Recombining channels 
        self.remap_label = None 

    def  __getitem__(self, idx):    
                
        ## Clear the buffer after selection?
        if idx == (len(self)-1): self.in_memory = []       

        if self.use_in_memory and len(self.in_memory)>0:
            im_cat, idx = self.in_memory 
        else:
            im_cat,idx = self.select_data_from_list(idx)
            self.in_memory = (im_cat,idx)


        inpt_im, targ_im = self.get_section(im_cat)
        inpt_im = torch.tensor(inpt_im).to(self.device).float()
        targ_im = torch.tensor(targ_im).to(self.device).float()  
        
        return (inpt_im, targ_im, idx) 

    def __len__(self):
        return self.selection_size

    def get_section(self, im_cat):

        inpt_im, targ_im = self.randomize_data( im_cat )
        
        if (inpt_im.dtype == np.uint8):   inpt_im = inpt_im/255.

        targ_im = self.generate_target(targ_im)

        return inpt_im, targ_im
        
    def generate_target(self, targ_im):

        if (targ_im.dtype == np.uint8) and (np.max(targ_im)>5):    
            targ_im = targ_im/255.

        if self.method == "any":
            targ_im = targ_im.max(axis=0,keepdims=True)
        
        if self.threshold is not None:
            targ_im = targ_im > self.threshold

            if self.area_range is not None:
                area_lims = self.area_range
                if len(area_lims)>1:
                    min_size,max_size = area_lims
                else:
                    min_size, max_size = area_lims, None
                targ_im = area_filter(targ_im, min_size=min_size, max_size=max_size)>0

            if self.imclose is not None:
                d = self.imclose                
                targ_im = bi_close(targ_im, radius=d )>0
                #targ_im = ndi.binary_closing(targ_im, structure=np.ones((1,d,d)) )

            if self.fill_area is not None:
                targ_im = area_filter(~targ_im, min_size=self.fill_area )==0

        if self.remap_label is not None:
            targ_im = relabel(targ_im,self.remap_label)

        return targ_im

    def select_data_from_list(self,idx=0):
        im_cat = None
        while im_cat is None:
            idx = self.select(idx)
            (fn, t) = self.fns[idx]
            im_cat = self.load_data(fn,t)
            if im_cat is None: 
                self.fns.pop(idx)
                self.selection_weights = np.delete(self.selection_weights, idx)
                idx = None

        return im_cat,idx

    def select(self, idx):
        maxN = len(self.selection_weights)
        if  0:#self.random_selection or idx is None:
            weights = self.selection_weights 
            weights = weights / np.sum(weights)
            idx = np.random.choice( maxN , 1, p = weights)[0]  
        else:
            idx = np.random.choice( maxN , 1)[0]  
        return idx

    def reset_weights(self):
        self.selection_weights = self.selection_weights*0 + 1

    def load_data(self, fn, t):
        return None

    def randomize_data(self, im_cat ):

        force_label = self.force_label 

        if self.randomize:

            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

            out_sz = np.array(self.outsize)
            crop_sz = (out_sz/scale).astype(np.int)
            
            im_cat = random_crop(im_cat, crop_sz, force_label )
            im_cat = [resize_to_base(im, size_out=out_sz) for im in im_cat]
            im_cat = random_flip(im_cat, self.rand_flip)
            
        else:
            out_sz = np.floor( np.array(im_cat[0].shape[1:3]) / 8)*8
            out_sz = out_sz.astype(np.int)  

        inpt_im = im_cat[0][:,:out_sz[0],:out_sz[1]]
        targ_im = im_cat[1][:,:out_sz[0],:out_sz[1]]     

        return  inpt_im, targ_im

class H5_Handler(Data_Handler):

    def __init__(self, fns, which="all", datasets=None, channels=None ):

        super().__init__()

        if datasets is None: datasets = ["image_data"]
        if channels is None: channels = [[0],[0]]

        if len(fns)==0: print("File list is empty")

        self.dataset_names = datasets
        self.channels = channels

        self.fns = make_file_list( fns, dataset=datasets[0], which=which)
        self.selection_weights = np.ones(len(self.fns))

    def load_data(self,fn,t):

        chans = self.channels
        dsets = self.dataset_names 

        if len(dsets) == 1:  dsets.append(dsets[0])
        ## Load Input
        input_im = read_h5( fn, dataset=dsets[0], t=t, c=chans[0]) 

        if input_im is None: return None
        if input_im.dtype == np.uint8: input_im = input_im/256.
        elif input_im.dtype == np.bool: input_im = input_im*1.

        ## Check if Input is the same as Target
        if (dsets[0] == dsets[1]) and (chans[0] == chans[1]):
            targt_im = input_im.copy()
            return (input_im, targt_im)

        ## Load Target
        targt_im = read_h5( fn, dataset=dsets[1], t=t, c=chans[1]) 
        if targt_im is None: ## Maybe do in a smarter way. 
            targt_im = read_h5( fn, dataset=dsets[1], t=0, c=chans[1]) 
            if targt_im is None: return None

        if targt_im.dtype == np.uint8:  
            if targt_im.max()>1:         targt_im = targt_im/256.
        elif targt_im.dtype == np.bool:  targt_im = targt_im*1.


        ## Set the images to the same size:
        sz_in = input_im[0].shape
        sz_tg = targt_im[0].shape
        if sz_in != sz_tg:
            targt_im = resize_to_base(targt_im, size_out=sz_in)
        
        return (input_im, targt_im)

class Tiff_Handler(Data_Handler):
    
    def __init__(self, targ_fns):
        
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inpt_fns = [fns.replace("label","input") for fns in targ_fns]

        self.fns = list(zip(inpt_fns,targ_fns))
        self.selection_weights = np.ones(len(self.fns))
    
    def load_data(self,idx):

        scale = 0.5
        inpt_fn, trgt_fn = self.fns[idx]

        inpt_im = np.array( io.imread( inpt_fn ))
        inpt_im = inpt_im.transpose((2,0,1))
        inpt_im = inpt_im/256
        inpt_im = resize_to_base( inpt_im, scale)
        
        targ_im = np.array( io.imread( trgt_fn ))
        targ_im = expand_label(targ_im, n_chans=2)
        targ_im = resize_to_base( targ_im, scale)

        return (inpt_im, targ_im)

### Helper Functions

def make_file_list(  file_names, dataset=None, which="all"  ):

    if dataset is None:   dataset = "image_data"

    file_list = []
    for fn in file_names:

        if isinstance(which, range):
            for n in which:  file_list.append( [fn,n] )
        elif which == "last":
                file_list.append( [fn,-1] )
        elif which == "first":
                file_list.append( [fn,0] )
        elif which == "all":
            with h5py.File( fn, 'r') as fh: 
                for n in range(len(fh[dataset])):  file_list.append( [fn,n] )

    return file_list

def read_h5(filename, dataset=None, t=None, c=None ):

    if dataset is None:    dataset = "image_data"

    #try:

    with h5py.File(filename, 'r') as fh:

        if dataset not in fh: return None
        data = fh[dataset]


        if fh[dataset].ndim == 3:           c = None
        elif c is not None: 
                if np.max(c) >= data.shape[1]: return None
        elif c is None:   c = slice(None)

        if fh[dataset].ndim == 2:           t = None
        elif t is not None: 
            #if np.max(t) >= data.shape[0]: return None
            if np.abs(t) >= data.shape[0]: return None
        elif t is None:   t = slice(None)
        
        
        if data.attrs.__contains__("active"):
            x1,x2,y1,y2 = data.attrs['active']
            x,y = slice(x1,x2), slice(y1,y2)
        else:
            x,y = slice(None), slice(None)

        if   fh[dataset].ndim == 4:          
            data = np.array(data[t,c,x,y])
        elif fh[dataset].ndim == 3:          
            data = np.array(data[t,x,y])
        elif fh[dataset].ndim == 2:           
            data = np.array(data[x,y])

        #if data.ndim == 3:           data = data[None,...]
        if data.ndim == 2:           data = data[None,:,:]

    return data

def expand_label(label_in, n_chans=None ):
    
    if n_chans is None:
        n_chans = label_in.max()
    label_im = np.stack([1.0*(label_in==i) for i in range(1,n_chans+1)])

    return label_im

def relabel(im,vmap):

    im = im.transpose(1,2,0)
    im = np.dot(im,vmap)
    im = im.transpose(2,0,1)
    return im 
    
def resize_to_base(im_in, reduction=None, base=1, size_out=None):
    
    if reduction==1:
        return im_in

    if size_out is None:
        sz_out = np.array(im_in.shape[1:3])[::-1] * reduction 
        sz_out = np.round(sz_out/base)*base
    else:
        if len(size_out)==3: size_out = size_out[1:]
        sz_out = size_out[::-1]
        sz_out = np.array(sz_out)

    #im_in = im_in.astype(np.uint8)
    sz_out = sz_out.astype(np.int)
    
    im_out = np.stack( [cv2.resize(im, tuple(sz_out)) for im in im_in ])
    return im_out

def random_crop(im, out_size, force_label=False):

    attempts = 10

    sz = np.array(im[0].shape)

    start = sz[1:3] - out_size + [1,1]
    start = np.maximum(start,1)
    for _ in range(attempts):
                
        s1 = np.random.randint(start[0])-1
        s2 = np.random.randint(start[1])-1
        start1 = np.maximum([s1,s2],0)
        end = (start1 + out_size).astype(np.int)

        ## Crop Label and check to see if there is anything there
        lb_crop = im[1][:,start1[0]:end[0], start1[1]:end[1]]
        if (not force_label) or (lb_crop).max(axis=0).mean() > 0.02 : break

    in_crop = im[0][:,start1[0]:end[0], start1[1]:end[1]]
    lb_crop = im[1][:,start1[0]:end[0], start1[1]:end[1]]
        
    im_out = (in_crop,lb_crop)
    return im_out

def random_flip(im_cat,prob):
    
    rand_var = np.random.rand(2)
    if rand_var[0]< prob[0]:
        im_cat = [ np.flip(im,axis=2) for im in im_cat]
    if rand_var[1]< prob[1]:
        im_cat = [ np.flip(im,axis=1) for im in im_cat]

    im_cat = [ im-0 for im in im_cat] 

    return im_cat

##########################

def bi_close(  bI,radius,down_sample=None):

    if down_sample is None:
        down_sample = minimum((5 / radius),1)
    
    im_size = np.array( bI.shape )

    d_size = (im_size*down_sample).astype(np.int)
    radius = int(radius * down_sample)

    filt = resize_to_base((bI*1.),  size_out=d_size) > 0.5
    dsk = mor.disk(radius)[None,:]
    filt = ndi.binary_closing( filt, dsk)
    bI_out = resize_to_base((filt*1.),  size_out=im_size) > 0.5

    return bI_out

def area_filter(ar, min_size=0, max_size=None):
    """
    """
    if ar.dtype == bool:
        ccs,l_max = ndi.label(ar)
    else:
        ccs = ar
        l_max = ar.max()

    component_sizes = np.bincount(ccs[ccs>0])
    idxs = np.arange(l_max+1).astype(np.uint16)
    if min_size>0:
        too_small = component_sizes < min_size
        idxs[too_small]=0

    if max_size is not None:
        too_large = component_sizes > max_size
        idxs[too_large]=0

    out = np.zeros_like(ccs, np.uint16)
    _, idxs2 = np.unique(idxs,return_inverse=True)
    out[ccs>0] = idxs2[ccs[ccs>0]]

    return out