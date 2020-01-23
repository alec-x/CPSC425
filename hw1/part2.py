import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import time

# Q1
def boxfilter(n):
    assert (n % 2) != 0, "Dimension \"n\" must be odd"
    elementNum = 1.0 / n**2
    return np.ones((n,n))*elementNum

# Q2
def roundOdd(n):
    assert n > 0, "input must be > 0"
    if (np.ceil(n) % 2) > 0:
        return int(np.ceil(n)) 
    else:
        return int(np.ceil(n)) + 1

def gauss1d(sigma):
    arrLen = roundOdd(sigma * 6)
    xDist = np.array(range(-arrLen/2 + 1, arrLen/2 + 1, 1)).astype(np.float)
    gaussNotNormal = np.exp(-xDist**2/(2*sigma**2))
    return gaussNotNormal/sum(gaussNotNormal)

# Q3
def gauss2d(sigma):
    dim1 = gauss1d(sigma)[np.newaxis]
    dim2 = np.transpose(dim1)
    convolution = convolve2d(dim1, dim2)
    return convolution/sum(convolution.flatten())

# Q4
def convolve2d_manual(array, filter):
    xLenArr, yLenArr = np.shape(array)
    xLenFilt, yLenFilt = np.shape(filter)
    array = np.pad(array,(xLenFilt/2,yLenFilt/2),'constant')
    convolvedArray = np.zeros((xLenArr,yLenArr))
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
    return convolve2d(array, gaussFilter)

def main():
    print("\nQuestion 1\n")
    for i in range(3,6):
        print "n = " + str(i)
        try:
            print boxfilter(i)
        except AssertionError as error:
            print(error)
            continue
    
    print("\nQuestion 2\n")
    for i in [0.3, 0.5, 1, 2]:
        print "sigma = " + str(i)
        try:
            print gauss1d(i)
        except AssertionError as error:
            print error
            continue

    print("\nQuestion 3\n")
    for i in [0.5, 1]:
        print "sigma = " + str(i)
        try:
            print gauss2d(i)
        except AssertionError as error:
            print error
            continue

    print("\nQuestion 4\n")
    
    im = Image.open('dog.jpg').convert('L')
    imArray = np.asarray(im, dtype=np.float32)
    
    q4Start = time.time()
    im2 = Image.fromarray(gaussconvolve2d_manual(imArray,3))
    q4Time = time.time() - q4Start
    im.show()
    im2.show()
    raw_input("press any key to continue...")
    
    print("\nQuestion 5\n")
    q5Start = time.time()
    im3 = Image.fromarray(gaussconvolve2d_scipy(imArray,3))
    q5Time = time.time() - q5Start
    im.show()
    im3.show()
    # raw_input("press any key to continue...")
    # The reason why correlation and convolution are the same here
    # is because correlation is the same as convolution with the matrix
    # turned 180 degrees. As the gaussian kernel is rotationally invariant
    # It makes no difference.
    print("\nQuestion 6\n")
    print "Manual convolve run time: " + str(q4Time)
    print "Scipy convolve run time: " + str(q5Time)

if __name__ == "__main__":
    main()