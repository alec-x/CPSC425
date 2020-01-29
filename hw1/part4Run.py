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
# Image.fromarray(speckFilteredMedian).show()
Image.fromarray(speckFilteredMedian).convert('RGB').save('.\\results\\p4q1a.png','PNG')

speckFilteredGauss = cv2.GaussianBlur(arrSpeck, ksize=(5,5), sigmaX=10, sigmaY=10)
# Image.fromarray(speckFilteredGauss).show()
Image.fromarray(speckFilteredGauss).convert('RGB').save('.\\results\\p4q1b.png','PNG')

speckFilteredBi = cv2.bilateralFilter(arrSpeck,15,110,30)
# Image.fromarray(speckFilteredBi).show()
Image.fromarray(speckFilteredBi).convert('RGB').save('.\\results\\p4q1c.png','PNG')
    
print("\nGaussian Noise filters\n")
GaussFilteredGauss = cv2.GaussianBlur(arrSpeck, ksize=(7,7), sigmaX= 30, sigmaY = 30)
# Image.fromarray(GaussFilteredGauss).show()
Image.fromarray(GaussFilteredGauss).convert('RGB').save('.\\results\\p4q1d.png','PNG')

GaussFilteredMedian = cv2.medianBlur(arrSpeck, 5)
# Image.fromarray(GaussFilteredMedian).show()
Image.fromarray(GaussFilteredMedian).convert('RGB').save('.\\results\\p4q1e.png','PNG')

GaussFilteredBi = cv2.bilateralFilter(arrSpeck,20,100,80)
# Image.fromarray(GaussFilteredBi).show()
Image.fromarray(GaussFilteredBi).convert('RGB').save('.\\results\\p4q1f.png','PNG')

raw_input("press any key to continue...")

print("\nQuestion 2\n")

print("\nSpeck Noise filters\n")
SpeckFilteredGauss = cv2.GaussianBlur(arrSpeck, ksize=(7, 7), sigmaX=50)
# Image.fromarray(SpeckFilteredGauss).show()
Image.fromarray(SpeckFilteredGauss).convert('RGB').save('.\\results\\p4q2a.png','PNG')

SpeckFilteredMedian = cv2.medianBlur(arrSpeck,5)
# Image.fromarray(SpeckFilteredMedian).show()
Image.fromarray(SpeckFilteredMedian).convert('RGB').save('.\\results\\p4q2b.png','PNG')

SpeckFilteredBi = cv2.bilateralFilter(arrSpeck, 7, sigmaColor=150, sigmaSpace=150)
# Image.fromarray(SpeckFilteredBi).show()
Image.fromarray(SpeckFilteredBi).convert('RGB').save('.\\results\\p4q2c.png','PNG')

print("\nGaussian Noise Filters\n")
GaussFilteredGauss = cv2.GaussianBlur(arrGauss, ksize=(7, 7), sigmaX=50)
# Image.fromarray(GaussFilteredGauss).show()
Image.fromarray(GaussFilteredGauss).convert('RGB').save('.\\results\\p4q2d.png','PNG')

GaussFilteredMedian = cv2.medianBlur(arrGauss,5)
# Image.fromarray(GaussFilteredMedian).show()
Image.fromarray(GaussFilteredMedian).convert('RGB').save('.\\results\\p4q2e.png','PNG')

GaussFilteredBi = cv2.bilateralFilter(arrGauss, 7, sigmaColor=150, sigmaSpace=150)
# Image.fromarray(GaussFilteredBi).show()
Image.fromarray(GaussFilteredBi).convert('RGB').save('.\\results\\p4q2f.png','PNG')

# In both types of noise, the median filter performed the best, although this may be due
# to the simple geometry. The median filter also introduced jagged edges to the shape.
# The gaussian filter was better at smoothing over the specks, but worse at preserving
# edges compared to the bilateral filter.