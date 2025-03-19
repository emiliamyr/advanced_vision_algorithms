import cv2
import numpy as np


# Ewaluacja wyników detekcji obiektów pierwszoplanowych
def zadanie4(filepath):
    TP, TN, FP, FN = 0, 0, 0, 0

    IG1 = cv2.imread(f'{filepath}/{filepath}/input/in000001.jpg', cv2.IMREAD_GRAYSCALE).astype("int")
    for i in range(300, 1099):
        IG = cv2.imread(f'{filepath}/{filepath}/input/in%06d.jpg' % (i + 1), cv2.IMREAD_GRAYSCALE).astype('int')
        I_true = cv2.imread(f'{filepath}/{filepath}/groundtruth/gt%06d.png' % (i + 1), cv2.IMREAD_GRAYSCALE).astype("int")
        IG_diff = cv2.absdiff(IG1, IG)
        IG_diff = 1 * (IG_diff > 7) * 255
        IG_diff = cv2.medianBlur(np.uint8(IG_diff), 15)
        IG_diff = cv2.morphologyEx(IG_diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        f = open(f'{filepath}/{filepath}/temporalROI.txt', 'r')  # open file
        line = f.readline()  # read line
        roi_start, roi_end = line.split()  # split line
        roi_start = int(roi_start)  # conversion to int
        roi_end = int(roi_end)  # conversion to int

        if roi_start < i < roi_end:
            TP_M = np.logical_and((IG_diff == 255), (I_true == 255))
            TP_S = np.sum(TP_M)
            TP += TP_S

            TN_M = np.logical_and((IG_diff == 0), (I_true == 0))
            TN_S = np.sum(TN_M)
            TN += TN_S

            FP_M = np.logical_and((IG_diff == 255), (I_true == 0))
            FP_S = np.sum(FP_M)
            FP += FP_S

            FN_M = np.logical_and((IG_diff == 0), (I_true == 255))
            FN_S = np.sum(FN_M)
            FN += FN_S

        cv2.imshow("I", np.uint8(IG_diff))
        cv2.waitKey(10)
        IG1 = IG

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    print(f"P = {P}, R = {R}, F1 = {F1}")

zadanie4("pedestrian") # P = 0.7468636599274235, R = 0.8160635123169891, F1 = 0.7799316466768574, thr = 7, mediana = 15
# zadanie4("office") # P = 0.7277854110078735, R = 0.33898604598769344, F1 = 0.4625341204758399, thr = 4, mediana = 15
# zadanie4("highway") # F1 = 0.8390607559323369, thr = 5, mediana = 9


