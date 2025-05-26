import cv2

I = cv2.imread('mandril.jpg')
height, width =I.shape[:2] # retrieving elements 1 and 2, i.e. the corresponding height and width
scale = 1.75 # scale factor
Ix2 = cv2.resize(I,(int(scale*height),int(scale*width)))
cv2.imshow("Big Mandrill",Ix2)
cv2.imshow("Mandril",I) # display
cv2.waitKey(0) # wait for key
cv2.destroyAllWindows() # close all windows

