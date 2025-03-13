import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

I = cv2.imread('lena.png')
I2 = cv2.imread('mandril.jpg')
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IG2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Lena",IG)
# cv2.imshow("Mandril",IG2)

ratio = 0.35
im_comb = cv2.addWeighted(IG, ratio, IG2, 1-ratio, 0)

img_add = cv2.add(IG.astype('uint16'), IG2.astype('uint16'))
im_mul = cv2.multiply(IG.astype('float64'), IG2.astype('float64'))
im_mul = (im_mul / np.max(im_mul) * 255)

cv2.imshow("Lena + Mandril", (img_add / 2).astype('uint8'))
cv2.imshow("Lena - Mandril", np.uint8((np.int16(IG) - np.int16(IG2) + 255)//2))
cv2.imshow("Lena * Mandril", im_mul.astype('uint8'))
cv2.imshow("| Lena - Mandril |", np.uint8(cv2.absdiff(IG, IG2)))
cv2.imshow("kombinacja liniowa", np.uint8(im_comb))

cv2.waitKey(0) # wait for key
cv2.destroyAllWindows() # close all windows
