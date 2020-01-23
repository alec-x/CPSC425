from part2Fun import *
import time

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
# The reason why correlation and convolution are the same here
# is because correlation is the same as convolution with the matrix
# flipped. As the gaussian kernel is rotationally invariant
# It makes no difference.

print("\nQuestion 6\n")
print "Manual convolve run time: " + str(q4Time)
print "Scipy convolve run time: " + str(q5Time)

print("\nQuestion 7\n")
# As the gaussian filter is separable, it would be more efficient to
# separate the filter into X and Y components and pass them separately through
# the image. This is due to the number of multiplications scaling O(2k)
# using two 1D arrays of length k, vs O(k^2) using one 2D array of length k