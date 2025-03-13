import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def hist(img):
    h = np.zeros((256, 1), np.float32)  # creates and zeros single-column arrays
    height, width = img.shape[:2]  # shape - we take the first 2 values
    for y in range(height):
        for x in range(width):
            pixel_value = img[y, x]
            h[pixel_value] += 1
    return h


I = cv2.imread('lena.png')
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

hist1 = cv2.calcHist([IG], [0], None, [256], [0, 256])
hist2 = hist(IG)
figLena, axsLena = plt.subplots(1, 3)
figLena.set_size_inches(20, 10)
axsLena[0].imshow(IG, 'gray', vmin=0, vmax=256)
axsLena[0].axis('off')
axsLena[1].plot(hist1)
axsLena[1].grid()
axsLena[2].plot(hist2)
axsLena[2].grid()
plt.show()
