import numpy as np
import cv2
import os
from os.path import join
import matplotlib . pyplot as plt


cap = cv2.VideoCapture('vid1_IR.avi')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # konwersja do skali szarości
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # binarzacja z progiem 35
    binary = cv2.threshold(G, 35, 255, cv2.THRESH_BINARY)[1]
    gauss = cv2.GaussianBlur(binary, (5, 5), 0)
    # element strukturalny
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # zamknięcie i otwarcie
    filtered = cv2.morphologyEx(gauss, cv2.MORPH_CLOSE, kernel)
    # filtered = cv2.morphologyEx(gauss, cv2.MORPH_OPEN, kernel)

    # indeksacja
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered)

    objects = []
    # iteracja po obiektach
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i] # współrzędne prostokąta otaczającego i pole powierzchni

        # sylwetka człowieka = powierzchnia prostokąta > 1000, pionowa sylwetka, minimalna wysokość
        if area > 1000 and h > w and h > 70:
            objects.append((x, y, w, h))
            # rysowanie prostokąta i etykieta obiektu
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Obj {i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # łączenie obiektów leżących pionowo jeden nad drugim
    # merged = []
    # used = set()
    #
    # for i, (x1, y1, w1, h1) in enumerate(objects):
    #     if i in used:
    #         continue
    #     merged_rect = (x1, y1, w1, h1)
    #
    #     for j, (x2, y2, w2, h2) in enumerate(objects):
    #         if i == j or j in used:
    #             continue
    #
    #     # czy prostokąty są blisko siebie w poziomie (mają podobne x)
    #     center_x1 = x1 + w1 / 2
    #     center_x2 = x2 + w2 / 2
    #     if abs(center_x1 - center_x2) > max(w1, w2):
    #         continue
    #
    #     # czy drugi prostokąt leży poniżej pierwszego i blisko w pionie
    #     if y2 > y1 and y2 < y1 + h1 * 1.5:
    #         # tworzenie prostokąta obejmującego oba
    #         x_min = min(x1, x2)
    #         y_min = min(y1, y2)
    #         x_max = max(x1 + w1, x2 + w2)
    #         y_max = max(y1 + h1, y2 + h2)
    #
    #         merged_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
    #         used.add(j)
    #
    #     used.add(i)
    #     merged.append(merged_rect)
    #
    # # wynikowe połączonych sylwetek
    # for x, y, w, h in merged:
    #   cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord ('q'): # break the loop when the ’q’ key is pressed
        break
cap.release()
cv2.destroyAllWindows()
