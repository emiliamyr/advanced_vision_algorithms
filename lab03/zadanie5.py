import cv2
import numpy as np


# KNN
KNN = cv2.createBackgroundSubtractorKNN(detectShadows=True)


def zadanie5(filepath):
    IG1 = cv2.imread(f'{filepath}/{filepath}/input/in000300.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(301, 1099):
        IG = cv2.imread(f'{filepath}/{filepath}/input/in%06d.jpg' % i, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        IG_diff = KNN.apply(IG)
        I_true = cv2.imread(f'{filepath}/{filepath}/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        f = open(f'{filepath}/{filepath}/temporalROI.txt', 'r')
        line = f.readline()
        roi_start, roi_end = line.split()
        roi_start = int(roi_start)
        roi_end = int(roi_end)

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

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    print(f"P = {P}, R = {R}, F1 = {F1}")


zadanie5("pedestrian")
# bez detekcji cieni
# P = 0.6164467903234484, R = 0.9223297787026243, F1 = 0.7389860791301606
# z detekcja cieni
# P = 0.49704637549385416, R = 0.8714222613751051, F1 = 0.6330247765592454
