from part1 import *
from PIL import Image
import numpy as np

# Part 1
'''
# Question 2 & 3
im = Image.open("dog.jpg").convert('RGB')
#im = Image.open("hw2part1//family.jpg").convert('L')
imArr = np.asarray(im, dtype=np.float32)
GPyramid = MakeGaussianPyramid(imArr, 0.75, 20)
ShowGaussianPyramid(GPyramid)
'''
# Question 4
template = Image.open("hw2part1/template.jpg").convert('L')
template = np.asarray(template, dtype=np.float32)

judyBatsIm = Image.open("hw2part1/judybats.jpg").convert('L')
judyBatsArr = np.asarray(judyBatsIm, dtype=np.float32)

pyramid = MakeGaussianPyramid(judyBatsArr, 0.75, 20)

templateMatch = FindTemplate(pyramid, template, 0.76, 15)