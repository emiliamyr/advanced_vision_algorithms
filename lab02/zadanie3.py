import cv2
import numpy as np

# Indeksacja i prosta analiza


IG1 = cv2.imread('pedestrian/pedestrian/input/in000001.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
for i in range(300, 1099):
    I = cv2.imread('pedestrian/pedestrian/input/in%06d.jpg' % (i + 1))
    I_VIS = I.copy()  # copy of the input image

    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    IG_diff = cv2.absdiff(IG1, IG)
    IG_diff = (IG_diff > 7).astype(np.uint8) * 255
    IG_diff = cv2.medianBlur(IG_diff, 5)
    IG_diff = cv2.morphologyEx(IG_diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(IG_diff)
    if stats.shape[0] > 1:  # are there any objects
        tab = stats[1:, 4]  # 4 columns without first element
        pi = np.argmax(tab)  # finding the index of the largest item
        pi = pi + 1  # increment because we want the index in stats , not in tab
        # drawing a bbox

        x, y, w, h, area = stats[pi]
        cv2.rectangle(I_VIS, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # print information about the field and the number of the largest element
        cv2.putText(I_VIS, f"Area: {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(I_VIS, f"ID: {pi}", (int(centroids[pi, 0]), int(centroids[pi, 1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Detected Objects", I_VIS)
    cv2.imshow(" Labels ", np.uint8(labels / retval * 255))
    cv2.waitKey(10)
    IG1 = IG
