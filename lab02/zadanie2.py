import cv2
import numpy as np

# Operacje morfologiczne

IG1 = cv2.imread('pedestrian/pedestrian/input/in000001.jpg', cv2.IMREAD_GRAYSCALE).astype('int')
for i in range(300, 1099):
    IG = cv2.imread('pedestrian/pedestrian/input/in%06d.jpg' % (i + 1), cv2.IMREAD_GRAYSCALE).astype('int')
    IG_diff = cv2.absdiff(IG1, IG)
    IG_diff = 1*(IG_diff > 10)*255
    IG_diff = cv2.medianBlur(np.uint8(IG_diff), 9)
    IG_diff = cv2.morphologyEx(IG_diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cv2.imshow("I", np.uint8(IG_diff))
    cv2.waitKey(10)
    IG1 = IG


