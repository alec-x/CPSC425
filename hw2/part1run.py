from part1 import *
from PIL import Image
import numpy as np

# Part 1
# Question 2 & 3

#im = Image.open("dog.jpg").convert('RGB')
im = Image.open("hw2part1//family.jpg").convert('L')
imArray = np.asarray(im, dtype=np.float32)
GPyramid = MakeGaussianPyramid(imArray, 0.75, 20)
ShowGaussianPyramid(GPyramid)