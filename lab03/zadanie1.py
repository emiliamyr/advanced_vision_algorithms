import cv2
import numpy as np


# Metody oparte o bufor próbek
def zadanie1(filepath, metoda):
    IG1 = cv2.imread(f'{filepath}/{filepath}/input/in000300.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    XX, YY = IG1.shape[:2]
    N = 60
    iN = 0
    BUF = np.zeros((XX, YY, N), np.uint8)
    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(301, 1099):
        IG = cv2.imread(f'{filepath}/{filepath}/input/in%06d.jpg' % i, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        I_true = cv2.imread(f'{filepath}/{filepath}/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        BUF[:, :, iN] = IG
        iN = (iN + 1) % N

        if metoda == "mediana":
            median = np.median(BUF, axis=2).astype(np.uint8)
            IG_diff = cv2.absdiff(median, IG)
        else:
            mean = np.mean(BUF, axis=2).astype(np.uint8)
            IG_diff = cv2.absdiff(mean, IG)

        IG_diff = 1*(IG_diff > 10)*255

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

# zadanie1("pedestrian")

# średnia: P = 0.07512148864062138, R = 0.9907253652763162, F1 = 0.1396537673307806
# mediana: P = 0.20353382450376137, R = 0.9932057561763821, F1 = 0.3378361831382577

zadanie1("office", "średnia")

# średnia: P = 0.5405815457462663, R = 0.6392012229390686, F1 = 0.5857695065750974
# mediana: P = 0.7012845123755828, R = 0.5545861449865813, F1 = 0.619367403764435
