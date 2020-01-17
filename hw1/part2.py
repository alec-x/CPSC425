import numpy as np


# Q1
def boxfilter(n):
    assert (n % 2) != 0, "Dimension \"n\" must be odd"
    elementNum = 1.0 / n**2
    return np.ones((n,n))*elementNum

print("Question 1")
for i in range(3,6):
    print "n = " + str(i)
    try:
        print(boxfilter(i))
    except AssertionError as error:
        print(error)
        continue

# Q2
