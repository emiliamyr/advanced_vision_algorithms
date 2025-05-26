import cv2
import numpy as np


# Aproksymacja sredniej i mediany
def zadanie2(filepath, metoda, a):
    IG1 = cv2.imread(f'{filepath}/{filepath}/input/in000300.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(301, 1099):
        IG = cv2.imread(f'{filepath}/{filepath}/input/in%06d.jpg' % i, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        I_true = cv2.imread(f'{filepath}/{filepath}/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        if metoda == "mediana":
            IG1[IG1 < IG] += 1
            IG1[IG1 > IG] -= 1

            IG_diff = cv2.absdiff(IG1, IG)
        else:
            mean = (a * IG + (1 - a) * IG1).astype(np.uint8)
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

zadanie2("pedestrian", "mediana", 0.01)

# mediana: P = 0.6716801586092139, R = 0.650439285575871, F1 = 0.6608890965529202
# średnia: P = 0.6466666762718873, R = 0.6690347642287088, F1 = 0.6576605816126699

# zadanie2("pedestrian", "średnia", 0.1)

# średnia: P = 0.6884894160016582, R = 0.633762234259803, F1 = 0.6599932629513552

# zadanie2("pedestrian", "średnia", 0.05)

# średnia: P = 0.6466666762718873, R = 0.6690347642287088, F1 = 0.6576605816126699
