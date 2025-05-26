import cv2
import numpy as np

prev_mask = None


# Polityka aktualizacji
def zadanie3(filepath, a):
    IG1 = cv2.imread(f'{filepath}/{filepath}/input/in000300.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(301, 1099):
        IG = cv2.imread(f'{filepath}/{filepath}/input/in%06d.jpg' % i, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        I_true = cv2.imread(f'{filepath}/{filepath}/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE).astype(np.uint8)


        IG_diff = cv2.absdiff(IG1, IG)

        IG_diff = 1*(IG_diff > 10)*255

        if IG_diff is not None:
            mask = np.logical_and(IG_diff == 0, prev_mask == 0)
            IG1[mask] = (a * IG[mask] + (1 - a) * IG1[mask]).astype(np.uint8)

        IG_diff = cv2.medianBlur(np.uint8(IG_diff), 5)
        IG_diff = cv2.morphologyEx(IG_diff, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        f = open(f'{filepath}/{filepath}/temporalROI.txt', 'r')  # open file
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
        IG1 = IG

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    print(f"P = {P}, R = {R}, F1 = {F1}")


zadanie3("office", 0.01)

# P = 0.619318441675836, R = 0.6862961384010422, F1 = 0.6510893206135152
