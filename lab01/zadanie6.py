import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

I = cv2.imread('lena.png')
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([IG], [0], None, [256], [0, 256])
IGE = cv2.equalizeHist(IG)
IGE_hist = cv2.calcHist([IGE], [0], None, [256], [0, 256])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# clipLimit - maximum height of the histogram bar - values above are distributed among neighbours
# tileGridSize - size of a single image block (local method, operates on separate image blocks)
I_CLAHE = clahe.apply(IG)
I_CLAHE_hist = cv2.calcHist([I_CLAHE], [0], None, [256], [0, 256])


figLena, axsLena = plt.subplots(2, 3)
figLena.set_size_inches(20, 10)
axsLena[0][0].imshow(IG, 'gray', vmin=0, vmax=256)
axsLena[0][0].axis('off')
axsLena[1][0].plot(hist)
axsLena[1][0].grid()
axsLena[0][1].imshow(IGE, 'gray', vmin=0, vmax=256)
axsLena[0][1].axis('off')
axsLena[1][1].plot(IGE_hist)
axsLena[1][1].grid()
axsLena[0][2].imshow(I_CLAHE, 'gray', vmin=0, vmax=256)
axsLena[0][2].axis('off')
axsLena[1][2].plot(I_CLAHE_hist)
axsLena[1][2].grid()
plt.show()
