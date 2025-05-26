import cv2
import numpy as np

# for i in range(300, 1100):
#     I = cv2.imread('pedestrian/pedestrian/input/in%06d.jpg' % i, cv2.IMREAD_GRAYSCALE)
#     cv2.imshow("I", I)
#     cv2.waitKey(10)

# Odejmowanie ramek i binaryzacja

IG1 = cv2.imread('pedestrian/pedestrian/input/in000001.jpg', cv2.IMREAD_GRAYSCALE).astype('int')
for i in range(300, 1099):
    IG = cv2.imread('pedestrian/pedestrian/input/in%06d.jpg' % (i + 1), cv2.IMREAD_GRAYSCALE).astype('int')
    IG_diff = cv2.absdiff(IG1, IG)
    IG_diff = 1*(IG_diff > 20)
    cv2.imshow("I", np.uint8(IG_diff*255))
    cv2.waitKey(10)
    IG1 = IG


