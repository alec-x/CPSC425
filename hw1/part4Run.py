from PIL import Image
import numpy as np
import math
from scipy import signal
import cv2

print("\nQuestion 1\n")

arrSpeck = np.asarray(Image.open('.\\part4images\\box_speckle.png'), dtype=np.float32) 
arrGauss = np.asarray(Image.open('.\\part4images\\box_gauss.png'), dtype=np.float32) 

print("\nSpeck Noise filters\n")
speckFilteredMedian = cv2.medianBlur(arrSpeck, 5)
Image.fromarray(speckFilteredMedian).show()

speckFilteredGauss = cv2.GaussianBlur(arrSpeck, ksize=(5,5), sigmaX=10, sigmaY=10)
Image.fromarray(speckFilteredGauss).show()

speckFilteredBi = cv2.bilateralFilter(arrSpeck,15,110,30)
Image.fromarray(speckFilteredBi).show()
    
print("\nGaussian Noise filters\n")
GaussFilteredGauss = cv2.GaussianBlur(arrSpeck, ksize=(7,7), sigmaX= 30, sigmaY = 30)
Image.fromarray(GaussFilteredGauss).show()

GaussFilteredMedian = cv2.medianBlur(arrSpeck, 5)
Image.fromarray(GaussFilteredMedian).show()

GaussFilteredBi = cv2.bilateralFilter(arrSpeck,20,100,80)
Image.fromarray(GaussFilteredBi).show()

raw_input("press any key to continue...")

print("\nQuestion 2\n")

print("\nSpeck Noise filters\n")
GaussFilteredGauss = cv2.GaussianBlur(arrSpeck, ksize=(7, 7), sigmaX=50)
Image.fromarray(GaussFilteredGauss).show()

GaussFilteredMedian = cv2.medianBlur(arrSpeck,5)
#Image.fromarray(GaussFilteredMedian).show()

GaussFilteredBi = cv2.bilateralFilter(arrSpeck, 7, sigmaColor=150, sigmaSpace=150)
Image.fromarray(GaussFilteredBi).show()

print("\nGaussian Noise Filters\n")
GaussFilteredGauss = cv2.GaussianBlur(arrGauss, ksize=(7, 7), sigmaX=50)
Image.fromarray(GaussFilteredGauss).show()

GaussFilteredMedian = cv2.medianBlur(arrGauss,5)
#Image.fromarray(GaussFilteredMedian).show()

GaussFilteredBi = cv2.bilateralFilter(arrGauss, 7, sigmaColor=150, sigmaSpace=150)
Image.fromarray(GaussFilteredBi).show()

# In both types of noise, the median filter performed the best, although this may be due
# to the simple geometry. The median filter also introduced jagged edges to the shape.
# The gaussian filter was better at smoothing over the specks, but worse at preserving
# edges compared to the bilateral filter.