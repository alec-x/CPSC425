from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal, ndimage
import ncc

# Part 1

# Question 2
# Assuming image as array
def MakeGaussianPyramid(image, scale, minsize):
    sig = 1.0/(2*scale)
    gaussianPyramid = []

    # make dtype consistent for consistent output list
    image = np.asarray(image, dtype=np.uint8)
    gaussianPyramid.append(image)
    
    # Check for if pyramid has stopped shrinking
    # Can initiate to 0 because we use equality to check if shrinking
    prevYLen = 0

    # Ensure we only retrieve x, y dims if RGB
    while (min(image.shape[0:2]) > minsize):
        yLen, xLen = image.shape[0:2]

        # If image isn't shrinking, terminate loop
        if(yLen == prevYLen):
            break
        
        # Gauss filter before resize. If RGB do by channel
        if (len(image.shape) == 2):
            image = ndimage.gaussian_filter(image.astype('float'), sigma=sig)
        else:
            image[:,:,0] = ndimage.gaussian_filter(image[:,:,0].astype('float'), sigma=sig)
            image[:,:,1] = ndimage.gaussian_filter(image[:,:,1].astype('float'), sigma=sig)
            image[:,:,2] = ndimage.gaussian_filter(image[:,:,2].astype('float'), sigma=sig)

        # Convert array to PIL image for resizing
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize((int(xLen*scale),int(yLen*scale)), Image.BICUBIC)
        image = np.asarray(image, dtype=np.float32)
        gaussianPyramid.append(image)
        
        prevYLen = yLen
    return gaussianPyramid

# Question 3
# Assume pyramid contains numpy arrays
def ShowGaussianPyramid(pyramid):
    # Only get the y and x dims in case input is RGB
    yLen, xLen = pyramid[0].shape[0:2]

    # Calculate total length of the pyramid display
    xLenTot = 0
    for im in pyramid:
        xLenTot = xLenTot + im.shape[1]
    
    # Initialize display area with white background
    pyrImg = Image.new("RGB", (xLenTot, yLen), color=(255,255,255))

    Offset = 0
    # Paste images into the display area
    for im in pyramid:
        pyrImg.paste(Image.fromarray(np.asarray(im,np.uint8)),(Offset, 0))
        Offset = Offset + im.shape[1]
        

    pyrImg.show()
    # pyrImg.convert('RGB').save('.\\results\\p1q3.png','PNG')
# Question 4

# Assuming pyramid, template as array. Casting threshold to double and templatewidth to int
def FindTemplate(pyramid, template, threshold, templateWidth):
    # y = dims[0], x = dims[1]. Find dimensions to scale to.
    # Even if RGB, dims[2] == 3, which should be smaller than x or y
    dims = list(template.shape)
    scaleFactor = float(max(dims)) / templateWidth
    yTempLen, xTempLen = np.multiply(dims, 1.0/scaleFactor)[0:2]

    # Apply gaussian before applying scaling. Turn back to PIL image for NCC
    template = ndimage.gaussian_filter(template.astype('float'), sigma=(1.0/2*scaleFactor))
    template = Image.fromarray(template.astype(np.uint8))
    template = template.resize((int(xTempLen),int(yTempLen)), Image.BICUBIC)

    # Figure out scale factor of pyramid, assuming pyramid has at least 2 elements.
    pyrScale = float(pyramid[0].shape[0])/pyramid[1].shape[0]
    currScale = 1
    # Array of x,y coords to hold highest scoring pixels
    pointsOfInterest = []
    scaleList = []

    for im in pyramid:
        im = Image.fromarray(im)
        # Points of interest in image scaled to original size 
        imPOI = np.argwhere(ncc.normxcorr2D(im, template) > threshold)
        imPOI = np.round(np.multiply(imPOI, currScale)).astype(int)

        # Append coordinates to coordinate list
        for coords in imPOI:
            pointsOfInterest.append(coords)
            scaleList.append([currScale*xTempLen, currScale*yTempLen])            
            
        # Modify scale
        currScale = pyrScale*currScale

    # Draw rectangles based on interesting points found
    im = Image.fromarray(pyramid[0]).convert('RGB')
    imDraw = ImageDraw.Draw(im)
    for i in range(0,len(pointsOfInterest)):
        yVal, xVal = pointsOfInterest[i]
        xLen, yLen = scaleList[i]
        xy = [xVal - xLen/2, yVal - yLen/2, xVal + xLen/2, yVal + yLen/2]
        imDraw.rectangle(xy,outline='red')
    
    im.show()

    return

# Part 2

# Question 2
def MakeLaplacianPyramid(image, scale, minsize):
    return