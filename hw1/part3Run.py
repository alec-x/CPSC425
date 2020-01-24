from hw1Functions import *

print("\nQuestion 1\n")
sigma = 6
imPath = '.\\part3images\\0a_cat.bmp'
arrBlur = gaussFilterColour(imPath, sigma)

imBlur = Image.fromarray(np.uint8(arrBlur))
# Image.open(imPath).show()
# imBlur.show()
# raw_input("press any key to continue...")

print("\nQuestion 2\n")
arrImage = np.asarray(Image.open(imPath), dtype=np.float32)
arrHigh =  arrImage - arrBlur + 128
imHigh = Image.fromarray(np.uint8(arrHigh))
# imHigh.show()


print("\nQuestion 3\n")
imPaths1 = ['0a_cat.bmp']
imPaths2 = ['0b_dog.bmp']
for i in range(0, len(imPaths1)):
    imHybrid = gaussHybridize('.\\part3images\\' + imPaths1[i], '.\\part3images\\' + imPaths2[i], 6)
    Image.fromarray(imHybrid).show()