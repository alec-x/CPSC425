from hw1Functions import *

print("\nQuestion 1\n")
# Show the gaussian blurred version of the image with sigma = 6
sigma = 6
imPath = '.\\part3images\\0a_cat.bmp'
arrBlur = gaussFilterColour(imPath, sigma)

imBlur = Image.fromarray(np.uint8(arrBlur))
Image.open(imPath).show()
imBlur.show()
raw_input("press any key to continue...")

print("\nQuestion 2\n")
# Show the high-pass filtered version of the image, adding 128 for visualization
arrImage = np.asarray(Image.open(imPath), dtype=np.float32)
arrHigh =  arrImage - arrBlur + 128
imHigh = Image.fromarray(np.uint8(arrHigh))
imHigh.show()
raw_input("press any key to continue...")

print("\nQuestion 3\n")

# Cat dog
# Show most interesting combinations found for hybrid images
imHybrid01 = gaussHybridize('.\\part3images\\0a_cat.bmp', '.\\part3images\\0b_dog.bmp' ,3,3)
Image.fromarray(imHybrid01).show()
imHybrid02 = gaussHybridize('.\\part3images\\0a_cat.bmp', '.\\part3images\\0b_dog.bmp' ,5,5)
Image.fromarray(imHybrid02).show()
imHybrid03 = gaussHybridize('.\\part3images\\0a_cat.bmp', '.\\part3images\\0b_dog.bmp' ,7,7)
Image.fromarray(imHybrid03).show()

# fish sub
imHybrid11 = gaussHybridize('.\\part3images\\3b_sub.bmp','.\\part3images\\3a_fish.bmp' ,3,3)
Image.fromarray(imHybrid11).show()
imHybrid12 = gaussHybridize('.\\part3images\\3b_sub.bmp','.\\part3images\\3a_fish.bmp' ,3,7)
Image.fromarray(imHybrid12).show()
imHybrid13 = gaussHybridize('.\\part3images\\3b_sub.bmp','.\\part3images\\3a_fish.bmp' ,5,3)
Image.fromarray(imHybrid13).show()


# einstein marilyn
imHybrid21 = gaussHybridize('.\\part3images\\2a_einstein.bmp','.\\part3images\\2b_marilyn.bmp' ,3,3)
Image.fromarray(imHybrid21).show()
imHybrid22 = gaussHybridize('.\\part3images\\2a_einstein.bmp','.\\part3images\\2b_marilyn.bmp' ,5,3)
Image.fromarray(imHybrid22).show()
imHybrid23 = gaussHybridize('.\\part3images\\2a_einstein.bmp','.\\part3images\\2b_marilyn.bmp' ,3,2)
Image.fromarray(imHybrid23).show()