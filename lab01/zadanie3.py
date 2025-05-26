import cv2

I = cv2.imread('mandril.jpg')
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
cv2.imshow("Mandril_IG",IG)
cv2.imshow("Mandril_IHSV", IHSV)
cv2.waitKey(0) # wait for key
cv2.destroyAllWindows() # close all windows
IH = IHSV[:,:,0]
IS = IHSV[:,:,1]
IV = IHSV[:,:,2]
print("IH = ", IH)
print("IS = ", IS)
print("IV = ", IV)

def rgb2gray(I):
    return 0.299*I[:,:,0] + 0.587*I[:,:,1] + 0.114*I[:,:,2]
