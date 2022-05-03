# Now write a function to loock at zstacks for plotting
import numpy as np
import matplotlib.pyplot as plt
# Set matplotlib backend
from ipywidgets import interact

def display_stack(image, step, cmap = "gray"):
    @interact(z=(0,image.shape[0],step))
    def select_z(z=1):
        # slice a plane
        plane = image[z,:,:]
        # Visualization
        plt.figure(figsize=(8,8))
        plt.imshow(plane, interpolation='none', cmap = cmap)
        plt.show()

# make a thing to show 2 image side by side
def display_stack2(image1, image2, step, cmap1 = "gray", cmap2 = "gray"):
    @interact(z=(0,image1.shape[0],step))
    def select_z(z=1):
        # slice a plane
        plane1 = image1[z,:,:]
        plane2 = image2[z,:,:]
        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(10,7))
        
             
        ax[0].imshow(plane1, interpolation='none', cmap = cmap1)
        ax[1].imshow(plane2, interpolation='none', cmap = cmap2)
        
        plt.show()
        
        
# Now write a function to layer z stacks over each other
def display_stack_layered(image1, image2, step, cmap1 = "gray", cmap2 = "gray"):
    @interact(z=(0,image1.shape[0],step))
    def select_z(z=1):
        # slice a plane
        plane1 = image1[z,:,:]
        plane2 = image2[z,:,:]
        # Visualization      
        plt.figure(figsize =[10, 10])
     
        plt.imshow(plane1, interpolation='none', cmap = cmap1)
        plt.imshow(plane2, interpolation='none', cmap = cmap2)
        
        plt.show()
        
# make some more convenience functions for plotting
# subset a stack and make projections
def project_stack_10(stack):
    tilelist = []   
    for i in range(1,11):
        # number of slices 
        nslice = stack.shape[0]
        # subset the slices of the loop
        nslice_loop = stack.shape[0]//10
        startslice = (i-1) * nslice_loop
        endslice = i*nslice_loop-1
        #print(startslice, endslice)
        
        #subset the image
        loopimg = stack[startslice:endslice,:,:]
        # calculate a maximum intesity prjection
        projection = np.max(loopimg, axis=0)
        tilelist.append(projection)
    return tilelist

def plot_ztile10(image, cmap = "gray"):
    tiledimage = project_stack_10(image)
    plt.figure(figsize=(20,7))
    for num, x in enumerate(tiledimage):
        loopimg = tiledimage[num]
        plt.subplot(2,5,num+1)
        plt.axis('off')
        plt.imshow(loopimg, cmap = cmap)    
        
# write this for an image + annotation layer array
def plot_ztile10_layered(image, annot, cmap1 = "gray", cmap2 = "prism"):
    tiledimage = project_stack_10(image)
    tiledannot = project_stack_10(annot)
    
    plt.figure(figsize=(20,7))
    for num, x in enumerate(tiledimage):
        loopimg = tiledimage[num]
        loopannot = tiledannot[num]
        
        plt.subplot(2,5,num+1)
        plt.axis('off')
        plt.imshow(loopimg, cmap = cmap1)
        plt.imshow(loopannot, cmap = cmap2, alpha = 0.5)

def mask_bg(binimage):
    mask = np.ma.array(binimage, mask = binimage == 0)
    return mask