import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

I = cv2.imread('lena.png')
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

gauss = cv2.GaussianBlur(IG, (5, 5), 0)
sobel = cv2.Sobel(IG, cv2.CV_32F, 1, 0)
laplasjan = cv2.Laplacian(IG, cv2.CV_32F)
mediana = cv2.medianBlur(IG, 5)

figLena, axsLena = plt.subplots(1, 5)
figLena.set_size_inches(20, 10)
axsLena[0].imshow(IG, 'gray', vmin=0, vmax=256)
axsLena[0].axis('off')
axsLena[0].set_title("Original")
axsLena[1].imshow(gauss, 'gray', vmin=0, vmax=256)
axsLena[1].axis('off')
axsLena[1].set_title("Gauss")
axsLena[2].imshow(sobel, 'gray', vmin=0, vmax=256)
axsLena[2].axis('off')
axsLena[2].set_title("Sobel")
axsLena[3].imshow(laplasjan, 'gray', vmin=0, vmax=256)
axsLena[3].axis('off')
axsLena[3].set_title("Laplacian")
axsLena[4].imshow(mediana, 'gray', vmin=0, vmax=256)
axsLena[4].axis('off')
axsLena[4].set_title("Median")
plt.show()
