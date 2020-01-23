from PIL import Image
import numpy as np
import math
from scipy import signal
import cv2

# Q1
def boxfilter(n):
    assert (n % 2) != 0, "Dimension \"n\" must be odd"
    elementNum = 1.0 / n**2
    return np.ones((n,n))*elementNum

# Q2
def roundOdd(n):
    assert n > 0, "input must be > 0"
    
    # Round to next odd integer
    # Determine if even by using modulus
    if (np.ceil(n) % 2) > 0:
        return int(np.ceil(n)) 
    else:
        return int(np.ceil(n)) + 1

def gauss1d(sigma):
    arrLen = roundOdd(sigma * 6)
    # Instantiate proper length 0-centered array
    xDist = np.array(range(-arrLen/2 + 1, arrLen/2 + 1, 1)).astype(np.float)
    # Find the un-normalized gaussian filter
    gaussNotNormal = np.exp(-xDist**2/(2*sigma**2))
    # Normalize gaussian filter
    return gaussNotNormal/sum(gaussNotNormal)

# Q3
def gauss2d(sigma):
    # Gaussian is separable and rotationally invariant. Therefore can 
    # turn 2d by convolving the 1D gaussian filter with its transpose
    dim1 = gauss1d(sigma)[np.newaxis]
    dim2 = np.transpose(dim1)
    convolution = signal.convolve2d(dim1, dim2)
    return convolution/sum(convolution.flatten())

# Q4
def convolve2d_manual(array, filter):
    # Unpadded length of array
    xLenArr, yLenArr = np.shape(array)
    # Size of filter
    xLenFilt, yLenFilt = np.shape(filter)
    # 0-Pad array wrt filter size
    array = np.pad(array,(xLenFilt/2,yLenFilt/2),'constant')
    # Instantiate output array of same size as input array
    convolvedArray = np.zeros((xLenArr,yLenArr))
    # Iterate through array and perform element-wise filter multiplication
    for i in range(0, xLenArr):
        for j in range (0, yLenArr):
            convolvedArray[i,j] = sum(np.multiply(filter, array[i:i+xLenFilt,j:j+yLenFilt]).flatten())
    return convolvedArray

def gaussconvolve2d_manual(array, sigma):
    gaussFilter = gauss2d(sigma)
    return convolve2d_manual(array, gaussFilter)

# Q5
def gaussconvolve2d_scipy(array, sigma):
    gaussFilter = gauss2d(sigma)
    return signal.convolve2d(array, gaussFilter)

