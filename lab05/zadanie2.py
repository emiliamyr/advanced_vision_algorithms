import cv2
import numpy as np
import matplotlib . pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# kalibracja układu kamer

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
# inner size of chessboard
width = 9
height = 6
square_size = 0.025 # 0.025 meters
img_width = 640
img_height = 480

# prepare object points , like (0 ,0 ,0) , (1 ,0 ,0) , (2 ,0 ,0) .... ,(8 ,6 ,0)
objp = np . zeros (( height * width , 1, 3), np.float64)
objp [: , 0, :2] = np.mgrid[0: width, 0: height].T.reshape ( -1 , 2)
objp = objp * square_size # Create real world coords . Use your metric .
# Arrays to store object points and image points from all the images .
objpoints = [] # 3d point in real world space
imgpointsLeft = [] # 2d points in image plane .
imgpointsRight = []
number_of_images = 50

for i in range(1, number_of_images):
    img_left = cv2.imread('pairs/pairs/left_%02d.png' % i)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

    img_right = cv2.imread('pairs/pairs/right_%02d.png' % i)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret_left and ret_right:
        objpoints.append(objp)

        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (3, 3), (-1, -1), criteria)
        imgpointsLeft.append(corners2_left)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (3, 3), (-1, -1), criteria)
        imgpointsRight.append(corners2_right)

        cv2.drawChessboardCorners(img_left, (width, height), corners2_left, ret_left)
        cv2.drawChessboardCorners(img_right, (width, height), corners2_right, ret_right)

        cv2.imshow('corners_left', img_left)
        cv2.imshow('corners_right', img_right)
        cv2.waitKey(100)
cv2.destroyAllWindows()

N_OK = len(objpoints)
K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))
K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))
rvecs_left = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rvecs_right = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_left = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_right = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

ret_left, K_left, D_left, _, _ = cv2.fisheye.calibrate(objpoints, imgpointsLeft, (img_width, img_height), K_left, D_left, rvecs_left, tvecs_left, calibration_flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

ret_right, K_right, D_right, _, _ = cv2.fisheye.calibrate(objpoints, imgpointsRight, (img_width, img_height), K_right, D_right, rvecs_right, tvecs_right, calibration_flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, np.eye(3), K_right, (img_width, img_height), cv2.CV_16SC2)
map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, np.eye(3), K_left, (img_width, img_height), cv2.CV_16SC2)

imgpointsLeft = np.asarray(imgpointsLeft, dtype=np.float64)
imgpointsRight = np.asarray(imgpointsRight, dtype=np.float64)

RMS, _, _, _, _, rotation_matrix, translation_vector = cv2.fisheye.stereoCalibrate(objpoints, imgpointsLeft, imgpointsRight, K_left, D_left, K_right, D_right, (img_width, img_height), None, None, cv2.CALIB_FIX_INTRINSIC, criteria)

R2 = np.zeros((3, 3))
P1 = np.zeros((3, 4))
P2 = np.zeros((3, 4))
Q = np.zeros((4, 4))

left_rectification, right_rectification, left_projection, right_projection, dispartity_to_depth_map = cv2.fisheye.stereoRectify(K_left, D_left, K_right, D_right, (img_width, img_height), rotation_matrix, translation_vector, 0, R2, P1, P2, Q, cv2.CALIB_ZERO_DISPARITY, (0, 0),0 ,0)

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, left_rectification, left_projection, (img_width, img_height), cv2.CV_16SC2)

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, right_rectification, right_projection, (img_width, img_height), cv2.CV_16SC2)


dst_left = cv2.remap(img_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
dst_right = cv2.remap(img_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)


N, XX, YY = dst_left.shape[::-1]

visRectify = np.zeros((YY, XX * 2, 3), np.uint8)
visRectify[:, :XX, :] = dst_left
visRectify[:, XX:, :] = dst_right

for y in range(0, YY, 10):
    cv2.line(visRectify, (0, y), (2 * XX, y), (255, 0, 0))

cv2.imshow('visRectify', visRectify)
cv2.waitKey(0)
cv2.destroyAllWindows()
