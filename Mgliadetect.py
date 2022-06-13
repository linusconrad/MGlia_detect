# Load libraries for file handling and image crunching
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

import scipy.ndimage as ndicpu
import cupyx.scipy.ndimage as ndi

import seaborn as sns
import pandas as pd

import cucim.skimage as skimage
import skimage as skimagecpu
# Import the os module
import os

#fancy gui viewer
#import napari

# progress bar for long computation
from tqdm import tqdm

# import own helper functions to subset and make boxes from coordinates
from boxhelpers_cp import *

# function to generate seeds from the microglia image 
def get_seeds(image, xystep, zstep, sigma=5, x=30):
    
    # run a gaussian filter on GPU
    filtered = ndi.filters.gaussian_filter(cp.asarray(image), sigma)
    
    # define a cube of x microns as a footprint
    # use the scale and floor division to find the number of pixels in each dimension to use
    foot = cp.ones((int(x//zstep),
               int(x//xystep),
               int(x//xystep)))
    
    locmax = skimage.feature.peak_local_max(cp.array(filtered), min_distance=0, footprint = foot)

    #preserve memory!
    del filtered

    #create an empty boolean array of the dimensions of the source img
    localhigh = cp.zeros_like(filtered, dtype=bool)

    # this will feed the coord to the empty mask
    localhigh[tuple(locmax.T)] = True
    
    # label the local highs 
    localhigh_img = ndi.label(localhigh)[0]
    print("found", np.unique(localhigh_img).shape[0], "seeds and", locmax.shape[0], " pixels")
    
    seedprops = skimage.measure.regionprops(localhigh_img, img)
    
    # preserve memory
    del localhigh 
    dellocalhigh_img 
    
    # loop through he object and get the seeds into an array
    seedlist = []
    for i in range(len(seedprops)):
        # make into np array of coordinates
        seed = np.array(seedprops[i].centroid).astype(int)
        seedlist.append(seed)
    
    return np.stack(seedlist)

# make a helper function to span a box around a 3d pixel coordinate
def seed_to_box(image, coords, npixels):
    # subset the box and set pixels to ones
    
    # the desired box gets spanned in 2 directions, we need to half this
    npixels = npixels//2
    # image boundaries
    boundaries = image.shape
    
    #print(boundaries)
    zstart = coords[0] - npixels
    zstop  = coords[0] + npixels
    
    xstart = coords[1] - npixels
    xstop  = coords[1] + npixels
    
    ystart = coords[2] -npixels
    ystop  = coords[2] + npixels
    # set fallback if image borders are touched
    if zstart < 0:
        zstart = 0
        
    if xstart < 0:
        xstart = 0
    
    if ystart < 0:
        ystart = 0
    
    # set fallback for end being larger than image boundaries
    if zstop > boundaries[0]:
        zstop = boundaries[0]
    
    if xstop > boundaries[1]:
        xstop = boundaries[1]
    
    if ystop > boundaries[2]:
        ystop = boundaries[2]
        
    box = np.zeros_like(image)
    # switch on pixels in the box
    box[zstart:zstop,xstart:xstop, ystart:ystop] = True 
    # push to mem and return
    return np.array(box).astype(bool)

# function to subset a box around a coordinate
def seed_to_subset(image, coords, npixels):
    # the desired box gets spanned in 2 directions, we need to half this
    npixels = npixels//2
    # image boundaries
    boundaries = image.shape
    
    #print(boundaries)
    zstart = coords[0] - npixels
    zstop  = coords[0] + npixels
    
    xstart = coords[1] - npixels
    xstop  = coords[1] + npixels
    
    ystart = coords[2] -npixels
    ystop  = coords[2] + npixels
    # set fallback if image borders are touched
    if zstart < 0:
        zstart = 0
        
    if xstart < 0:
        xstart = 0
    
    if ystart < 0:
        ystart = 0
    
    # set fallback for end being larger than image boundaries
    if zstop > boundaries[0]:
        zstop = boundaries[0]
    
    if xstop > boundaries[1]:
        xstop = boundaries[1]
    
    if ystop > boundaries[2]:
        ystop = boundaries[2]
        
    # subset the image + return
    imgbox = image[zstart:zstop,xstart:xstop, ystart:ystop]
    return imgbox

# define a function that makes a local threshold from a subsetted box
def find_cell_thresh(image, seed, expandpix):
    # subset the ROI and calulate thresh based on ROI
    ROI = seed_to_subset(image, seed, expandpix)
    Thresh = skimagecpu.filters.threshold_otsu(ROI) 
    return Thresh

# define a function that makes a cell from threshold
def detect_cell_thresh(image, seedcoord, thresh):
    # push to GPU + create a binary image
    image = cp.asarray(image)
    bin_img = image > thresh
    # floodfill the detected cell
    floodseed = tuple((seedcoord[0],seedcoord[1],seedcoord[2]))
    
    bin_img_cpu = bin_img.get()
    
    cellimg = skimagecpu.segmentation.flood(bin_img_cpu,floodseed)
    
    return np.array(cellimg).astype(bool)


# iterative cell detection
def detect_cell_iter(image, seedcoord, expandpix, vlow, vhigh):
    
    # setup a threshold for the iterating, start with a little less than Otsu
    # this way it reduces the volume from a too large fit
    thresh_iter = find_cell_thresh(image, seedcoord, expandpix)*0.6
    void_mask = cp.zeros_like(image)
    
    # get the candidate cell mask
    CCM = detect_cell_thresh(image, seedcoord, thresh_iter)
    # count the number of pixels in the mask (volume)
    vol = cp.count_nonzero(cp.asarray(CCM))
    
    n_tries = 1
    # if the volume is within the tolerance, return the mask
    if (vol < vhigh and vol >vlow):
        #print("...done in one go")
        return CCM
    
    # while the number of pixels is outside the tolerance
    while not(vol < vhigh and vol >vlow):
        # if the volume is larger than target interval set threshold to previous*1.x
        if vol > vhigh:
            #print("too large")
            thresh_iter = thresh_iter*1.2
            n_tries = n_tries + 1
            CCM = detect_cell_thresh(image, seedcoord, thresh_iter)
            #update volume
            vol = cp.count_nonzero(cp.asarray(CCM))
            
        # if the volume is below target interval set threshold to previous*0.x
        if vol < vlow:
            #print("too small")
            thresh_iter = thresh_iter*0.8
            n_tries = n_tries + 1
            CCM = detect_cell_thresh(image, seedcoord, thresh_iter)
            #update volume
            vol = cp.count_nonzero(cp.asarray(CCM))
            
        # if the number of iterations is high and the cellmask is tiny than an absolute minimum, break and return empty mask
        if (vol < vlow and n_tries > 6):
            #print("Bad seed: Just a specle")
            return void_mask.get()
        
        # if the number of iterations is high and the cell mask is massive, the seed is on the bg, break and return empty mask
        if (vol > vhigh*3 and n_tries > 4):
            #print("Bad seed: bg pixel, memory used: ", getGPUmem())
            return void_mask.get()
        
        # if a reasonable volume is found return it
        if (vol < vhigh and vol >vlow):
            #print("...Found mask in", n_tries, "iterations")
            return CCM
        
        # if no solution is found (bouncing right between to high and too low)
        if (n_tries > 6):
            #print("...Bad seed: Cant find a solution")
            return void_mask.get()      