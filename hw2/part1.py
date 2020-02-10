from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal, ndimage
import ncc

# Part 1

# Assuming image as array

# Question 2
def MakeGaussianPyramid(image, scale, minsize):
    sig = 1.0/(2*scale)
    gaussianPyramid = []
    # make sure dtype is correct
    image = np.asarray(image, dtype=np.float32)
    gaussianPyramid.append(image)
    while (min(image.shape) > minsize):
        yLen, xLen = image.shape
        image = ndimage.gaussian_filter(image, sigma=sig)
        image = Image.fromarray(image)
        image = image.resize((int(xLen*scale),int(yLen*scale)), Image.BICUBIC)
        image = np.asarray(image, dtype=np.float32)
        gaussianPyramid.append(image)
        
    return gaussianPyramid

# Question 3
def ShowGaussianPyramid(pyramid):
    
    yLen, xLen = pyramid[0].shape
    xLenTot = 0
    for im in pyramid:
        xLenTot = xLenTot + im.shape[1]
    
    pyrImg = Image.new("RGB", (xLenTot, yLen), color=(255,255,255))

    Offset = 0
    for im in pyramid:
        pyrImg.paste(Image.fromarray(im),(Offset, 0))
        Offset = Offset + im.shape[1]
        

    pyrImg.show()

# Question 4

# Assuming pyramid, template as array. Casting threshold to double and templatewidth to int
def FindTemplate(pyramid, template, threshold, templateWidth):
    # y = dims[0], x = dims[1]. Find dimensions to scale to.
    dims = list(template.shape)
    scaleFactor = max(dims) / templateWidth
    yTempLen, xTempLen = np.multiply(dims, 1.0/scaleFactor)
    # Apply gaussian before applying scaling. Turn back to PIL image for NCC
    template = ndimage.gaussian_filter(template, sigma=(1.0/2*scaleFactor))
    template = Image.fromarray(template.resize((int(xTempLen),int(yTempLen)), Image.BICUBIC))

    # Figure out scale factor of pyramid, assuming pyramid has at least 2 elements.
    pyrScale = pyramid[0].shape[0]/pyramid[1].shape[0]
    # Array of tuples to hold highest scoring pixels
    POI = []
    for im in pyramid:
        im = Image.fromarray(im)
        # Points of interest in image scaled to original size 
        imPOI = np.argwhere(ncc.normxcorr2D(im, template) > threshold)
        POI.append()
