import cv2

I = cv2.imread('mandril.jpg')
cv2.imwrite("mandril.png",I) # zapis obrazu do pliku
cv2.imshow("Mandril",I) # display
cv2.waitKey(0) # wait for key
cv2.destroyAllWindows() # close all windows
print(I.shape) # dimensions /rows, columns, depth/
print(I.size) # number of bytes
print(I.dtype) # data type
