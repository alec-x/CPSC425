from hw2functions import *
from PIL import Image
import numpy as np
# Part 2

# Question 2 & 3
imOrchid = Image.open(".//hw2part2//orchid.jpg").convert('RGB')
imViolet = Image.open(".//hw2part2//violet.jpg").convert('RGB')

imOrchidArr = np.asarray(imOrchid, dtype=np.float32)
imVioletArr = np.asarray(imViolet, dtype=np.float32)

# Define uniform scale and minsize
scale = 0.75
minsize = 20

LPyramidOrchid = MakeLaplacianPyramid(imOrchidArr, scale, minsize)
LPyramidViolet = MakeLaplacianPyramid(imVioletArr, scale, minsize)

ShowLaplacianPyramid(LPyramidOrchid)
ShowLaplacianPyramid(LPyramidViolet)

# Question 4
ShowGaussianPyramid(ReconstructGaussianFromLaplacian(LPyramidOrchid))
ShowGaussianPyramid(ReconstructGaussianFromLaplacian(LPyramidViolet))

# Question 5
imMask = Image.open(".//hw2part2//orchid_mask.bmp").convert('RGB')
imMaskArr = np.asarray(imMask, dtype=np.float32)

GPyramidMask = MakeGaussianPyramid(imMaskArr, scale, minsize)
ShowGaussianPyramid(GPyramidMask)


# Question 6
# Make minsize smaller for better blends
minsize = 8

LPyramidOrchid = MakeLaplacianPyramid(imOrchidArr, scale, minsize)
LPyramidViolet = MakeLaplacianPyramid(imVioletArr, scale, minsize)
GPyramidMask = MakeGaussianPyramid(imMaskArr, scale, minsize)

LPyramidBlend1 = BlendPyramids(LPyramidOrchid,LPyramidViolet,GPyramidMask)
ShowGaussianPyramid(ReconstructGaussianFromLaplacian(LPyramidBlend1))

# Question 7

imMask = np.asarray(Image.open(".//hw2part2//tomato_mask.bmp").convert('RGB'), dtype=np.float32)
im1 = np.asarray(Image.open(".//hw2part2//tomato.jpg").convert('RGB'), dtype=np.float32)
im2 = np.asarray(Image.open(".//hw2part2//apple.jpg").convert('RGB'), dtype=np.float32)

im1 = MakeLaplacianPyramid(im1, scale, minsize)
im2 = MakeLaplacianPyramid(im2, scale, minsize)
imMask = MakeGaussianPyramid(imMask, scale, minsize)

LPyramidBlend2 = BlendPyramids(im2,im1,imMask)
ShowGaussianPyramid(ReconstructGaussianFromLaplacian(LPyramidBlend2))

imMask = np.asarray(Image.open(".//hw2part2//blue_cup.jpg").convert('RGB'), dtype=np.float32)
im1 = np.asarray(Image.open(".//hw2part2//green_cup.jpg").convert('RGB'), dtype=np.float32)
im2 = np.asarray(Image.open(".//hw2part2//cup_mask.bmp").convert('RGB'), dtype=np.float32)

im1 = MakeLaplacianPyramid(im1, scale, minsize)
im2 = MakeLaplacianPyramid(im2, scale, minsize)
imMask = MakeGaussianPyramid(imMask, scale, minsize)

LPyramidBlend3 = BlendPyramids(im2,im1,imMask)
ShowGaussianPyramid(ReconstructGaussianFromLaplacian(LPyramidBlend3))