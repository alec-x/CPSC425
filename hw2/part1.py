from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc

# Assuming image as PIL Image
def MakeGaussianPyramid(image, scale, minsize):
    xLen, yLen = image.size
    sig = 1.0/(2*scale)
    gaussianPyramid = []
    while (min(xLen, yLen >= minsize):
        image = scipy.ndimage.gaussian_filter(image, sigma=sig)
        gaussianPyramid.append(image)
        image = image.resize((int(x*scale),int(y*scale)), Image.BICUBIC)
        xLen, yLen = image.size
    return gaussianPyramid

def ShowGaussianPyramid(pyramid):
    
    xLenTot = int(float(n)*XLen/(1-1/2^n)) # Using solution to geometric series 
    yLenTot = int(float(n)*YLen/(1-1/2^n))
    xOffset = 0
    yOffset = 0
    pyrImg = Image.new("L", (xLenTot, yLenTot))

    for im in pyramid:
        xLen, yLen = im.size
        xOffset = xOffset + math.ceil(xLen/2)
        yOffset = yOffset + math.ceil(yLen/2)
        pyrImg.paste(im,(xOffset, yOffset))

    pyrImg.show()