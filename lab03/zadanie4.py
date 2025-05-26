import cv2
import numpy as np


# GMM/MOG
MOG = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=50, detectShadows=True)


def zadanie4(filepath, a):
    IG1 = cv2.imread(f'{filepath}/{filepath}/input/in000300.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(301, 1099):
        IG = cv2.imread(f'{filepath}/{filepath}/input/in%06d.jpg' % i, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        IG_diff = MOG.apply(IG)
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


zadanie4("pedestrian", 0.01)
# bez detekcji cieni
# P = 0.7716051272630784, R = 0.6762031535546804, F1 = 0.7207609284560105\

# z detekcja cieni
# P = 0.9549901726060941, R = 0.5475322078381953, F1 = 0.6960134297848904
