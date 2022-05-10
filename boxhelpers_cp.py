import cupy as cp

# define a function that spans the bounding box around the cell and makes a mask image with the box
def coord_to_box(image, coords):
    box = cp.zeros_like(image)
    # subset the box and set pixels to ones
    box[coords[1]:coords[4],coords[2]:coords[5],coords[3]:coords[6]] = True
    return cp.array(box).astype(bool)

def coord_to_box_expand(image, coords, npixels):
    # subset the box and set pixels to ones
 
    # image boundaries
    boundaries = image.shape
    #print(boundaries)
    zstart = coords[1] - npixels
    zstop  = coords[4] + npixels
    xstart = coords[2] -npixels
    xstop  = coords[5] + npixels
    ystart = coords[3] -npixels
    ystop  = coords[6] + npixels
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
        
    box = cp.zeros_like(image)
    # switch on pixels in the box
    box[zstart:zstop,xstart:xstop, ystart:ystop] = True 
    return cp.array(box).astype(bool)

# return  o box selection from an image
def subsetbox(image, coords):
    # subset the box and set pixels to ones
    return image[coords[1]:coords[4],coords[2]:coords[5],coords[3]:coords[6]]

# return a subsetted ROI (box) around a pixel 
def subsetbox_expand(image, coords, npixels):
    # image boundaries
    boundaries = image.shape
    #print(boundaries)
    zstart = coords[1] - npixels
    zstop  = coords[4] + npixels
    xstart = coords[2] - npixels
    xstop  = coords[5] + npixels
    ystart = coords[3] - npixels
    ystop  = coords[6] + npixels
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
        
    return image[zstart:zstop,xstart:xstop, ystart:ystop]