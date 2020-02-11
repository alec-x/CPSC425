from part1 import *
from PIL import Image
import numpy as np

# Part 1

# Question 2 & 3
im = Image.open("dog.jpg").convert('RGB')
#im = Image.open("hw2part1//family.jpg").convert('L')
imArr = np.asarray(im, dtype=np.float32)
GPyramid = MakeGaussianPyramid(imArr, 0.75, 20)
ShowGaussianPyramid(GPyramid)

# Question 4
template = Image.open("hw2part1/template.jpg").convert('L')
template = np.asarray(template, dtype=np.float32)

judyBatsIm = Image.open("hw2part1/judybats.jpg").convert('L')
judyBatsArr = np.asarray(judyBatsIm, dtype=np.float32)

pyramid = MakeGaussianPyramid(judyBatsArr, 0.75, 20)

FindTemplate(pyramid, template, 0.76, 15)

# Question 5
# Using 0.70 as the threshold, we get 15 missed faces and 3 not faces
# Using 0.69 as the threshold, we get 14 missed faces and 11 not faces
# Using 0.68 as the threshold, we get 11 missed faces and 17 not faces
template = Image.open("hw2part1/template.jpg").convert('L')
template = np.asarray(template, dtype=np.float32)

Arrs = []
imgNames = ["judybats","students","tree","family","fans","sports"]
for i in range(0, len(imgNames)):
    img = Image.open("hw2part1/" + imgNames[i] + ".jpg").convert('L')
    Arrs.append(img)

pyramids = []
for i in range(0, len(Arrs)):
    pyramids.append(MakeGaussianPyramid(Arrs[i], 0.75, 20))
    FindTemplate(pyramids[i], template, 0.69, 15)

# Question 6
# judybats: 4/5 = 80%
# students: 19/27 = 70% 
# tree: 0/0 = 100%?
# family: 0/3 = 0%
# fans: 0/3 = 0%
# sports: 0/1 = 0%
# As the template describes a white, young-adult, male at a frontal angle, with overhead lighting
# it seems that the algorithm is best at detecting that. For most of the
# unrecognized faces, the face was at a different angle, different lighting, female, or different ethnicity
