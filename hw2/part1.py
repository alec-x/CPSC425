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
def FindTemplate(pyramid, template, threshold):
    pass