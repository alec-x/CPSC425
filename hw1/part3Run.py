from hw1Functions import *

print("\nQuestion 1\n")
# Show the gaussian blurred version of the image with sigma = 6
sigma = 6
imPath = '.\\part3images\\0b_dog.bmp'
arrBlur = gaussFilterColour(imPath, sigma)

imBlur = Image.fromarray(np.uint8(arrBlur))
# Image.open(imPath).show()
Image.open(imPath).convert('RGB').save('.\\results\\p3q1a.png','PNG')
# imBlur.show()
imBlur.convert('RGB').save('.\\results\\p3q1b.png','PNG')

raw_input("press any key to continue...")

print("\nQuestion 2\n")

imPath = '.\\part3images\\0a_cat.bmp'
arrBlur = gaussFilterColour(imPath, sigma)

# Show the high-pass filtered version of the image, adding 128 for visualization
arrImage = np.asarray(Image.open(imPath), dtype=np.float32)
arrHigh =  arrImage - arrBlur + 128
imHigh = Image.fromarray(np.uint8(arrHigh))
# imHigh.show()
imHigh.convert('RGB').save('.\\results\\p3q2.png','PNG')
raw_input("press any key to continue...")

print("\nQuestion 3\n")

# Cat dog
# Show most interesting combinations found for hybrid images
imHybrid01 = gaussHybridize('.\\part3images\\0a_cat.bmp', '.\\part3images\\0b_dog.bmp' ,3,3)
# Image.fromarray(imHybrid01).show()
Image.fromarray(imHybrid01).convert('RGB').save('.\\results\\p3q3a.png','PNG')
imHybrid02 = gaussHybridize('.\\part3images\\0a_cat.bmp', '.\\part3images\\0b_dog.bmp' ,5,5)
# Image.fromarray(imHybrid02).show()
Image.fromarray(imHybrid02).convert('RGB').save('.\\results\\p3q3b.png','PNG')
imHybrid03 = gaussHybridize('.\\part3images\\0a_cat.bmp', '.\\part3images\\0b_dog.bmp' ,7,7)
# Image.fromarray(imHybrid03).show()
Image.fromarray(imHybrid03).convert('RGB').save('.\\results\\p3q3c.png','PNG')

# fish sub
imHybrid11 = gaussHybridize('.\\part3images\\3b_sub.bmp','.\\part3images\\3a_fish.bmp' ,3,3)
# Image.fromarray(imHybrid11).show()
Image.fromarray(imHybrid11).convert('RGB').save('.\\results\\p3q3d.png','PNG')
imHybrid12 = gaussHybridize('.\\part3images\\3b_sub.bmp','.\\part3images\\3a_fish.bmp' ,3,7)
# Image.fromarray(imHybrid12).show()
Image.fromarray(imHybrid12).convert('RGB').save('.\\results\\p3q3e.png','PNG')
imHybrid13 = gaussHybridize('.\\part3images\\3b_sub.bmp','.\\part3images\\3a_fish.bmp' ,5,3)
# Image.fromarray(imHybrid13).show()
Image.fromarray(imHybrid13).convert('RGB').save('.\\results\\p3q3f.png','PNG')


# einstein marilyn
imHybrid21 = gaussHybridize('.\\part3images\\2a_einstein.bmp','.\\part3images\\2b_marilyn.bmp' ,3,3)
# Image.fromarray(imHybrid21).show()
Image.fromarray(imHybrid21).convert('RGB').save('.\\results\\p3q3g.png','PNG')
imHybrid22 = gaussHybridize('.\\part3images\\2a_einstein.bmp','.\\part3images\\2b_marilyn.bmp' ,5,3)
# Image.fromarray(imHybrid22).show()
Image.fromarray(imHybrid22).convert('RGB').save('.\\results\\p3q3h.png','PNG')
imHybrid23 = gaussHybridize('.\\part3images\\2a_einstein.bmp','.\\part3images\\2b_marilyn.bmp' ,3,2)
# Image.fromarray(imHybrid23).show()
Image.fromarray(imHybrid23).convert('RGB').save('.\\results\\p3q3i.png','PNG')