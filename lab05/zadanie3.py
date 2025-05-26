import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

width = 9
heigth = 6
square_size = 0.025

objp = np.zeros((heigth * width, 1, 3), np.float64)
objp[:, 0, :2] = np.mgrid[0:width, 0:heigth].T.reshape(-1, 2)

objp = objp * square_size

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

img_width = 640
img_height = 480

number_of_images = 50
for i in range(1, number_of_images):
    img = cv2.imread('pairs/pairs/left_%02d.png' % i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (width, heigth),
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    Y, X, channels = img.shape

    if ret:
        minRx = corners[:, :, 0].min()
        maxRx = corners[:, :, 0].max()
        minRy = corners[:, :, 1].min()
        maxRy = corners[:, :, 1].max()

        border_threshold_x = X / 12
        border_threshold_y = Y / 12

        x_thresh_bad = False
        if minRx < border_threshold_x:
            x_thresh_bad = True

        y_thresh_bad = False
        if minRy < border_threshold_y:
            y_thresh_bad = True

        if y_thresh_bad or x_thresh_bad:
            continue

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, (width, heigth), corners2, ret)


N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

ret, K, D, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints, (img_width, img_height), K, D, rvecs, tvecs,
                                        calibration_flags,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (img_width, img_height), cv2.CV_16SC2)

img = cv2.imread('pairs/pairs/left_48.png')
undistorted_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

width = 9
heigth = 6
square_size = 0.025

objp = np.zeros((heigth * width, 1, 3), np.float64)
objp[:, 0, :2] = np.mgrid[0:width, 0:heigth].T.reshape(-1, 2)

objp = objp * square_size

objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane.
imgpoints_right = []  # 2d points in image plane.

for i in range(1, number_of_images):
    img_left = cv2.imread('pairs/pairs/left_%02d.png' % i)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

    img_right = cv2.imread('pairs/pairs/right_%02d.png' % i)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (width, heigth),
                                                       cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (width, heigth),
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret_left and ret_right:
        objpoints.append(objp)

        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (3, 3), (-1, -1), criteria)
        imgpoints_left.append(corners2_left)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (3, 3), (-1, -1), criteria)
        imgpoints_right.append(corners2_right)

        cv2.drawChessboardCorners(img_left, (width, heigth), corners2_left, ret_left)
        cv2.drawChessboardCorners(img_right, (width, heigth), corners2_right, ret_right)



N_OK = len(objpoints)
K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))
K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))
rvecs_left = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rvecs_right = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_left = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_right = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

ret_left, K_left, D_left, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints_left, (img_width, img_height), K_left,
                                                       D_left, rvecs_left, tvecs_left, calibration_flags,
                                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

ret_right, K_right, D_right, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints_right, (img_width, img_height), K_right,
                                                          D_right, rvecs_right, tvecs_right, calibration_flags, (
                                                          cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, np.eye(3), K_right,
                                                             (img_width, img_height), cv2.CV_16SC2)
map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, np.eye(3), K_left, (img_width, img_height),
                                                           cv2.CV_16SC2)

imgpoints_left = np.asarray(imgpoints_left, dtype=np.float64)
imgpoints_right = np.asarray(imgpoints_right, dtype=np.float64)

RMS, _, _, _, _, rotation_matrix, translation_vector = cv2.fisheye.stereoCalibrate(objpoints, imgpoints_left,
                                                                                   imgpoints_right, K_left, D_left,
                                                                                   K_right, D_right,
                                                                                   (img_width, img_height), None, None,
                                                                         cv2.CALIB_FIX_INTRINSIC, criteria)

R2 = np.zeros((3, 3))
P1 = np.zeros((3, 4))
P2 = np.zeros((3, 4))
Q = np.zeros((4, 4))

left_rectification, right_rectification, left_projection, right_projection, dispartity_to_depth_map = cv2.fisheye.stereoRectify(
    K_left, D_left, K_right, D_right, (img_width, img_height), rotation_matrix, translation_vector, 0, R2, P1, P2, Q,
    cv2.CALIB_ZERO_DISPARITY, (0, 0), 0, 0)

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, left_rectification, left_projection,
                                                           (img_width, img_height), cv2.CV_16SC2)

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, right_rectification, right_projection,
                                                             (img_width, img_height), cv2.CV_16SC2)

dst_left = cv2.remap(img_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
dst_right = cv2.remap(img_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

N, XX, YY = dst_left.shape[::-1]

visRectify = np.zeros((YY, XX * 2, 3), np.uint8)
visRectify[:, :XX, :] = dst_left
visRectify[:, XX:, :] = dst_right

#####################
img = cv2.imread("example/example/example0.jpg")
h, w, _ = img.shape
left = img[:, :w // 2]
right = img[:, w // 2:]

left = cv2.resize(left, (map1_left.shape[1], map1_left.shape[0]))
right = cv2.resize(right, (map1_right.shape[1], map1_right.shape[0]))

undist_left = cv2.remap(left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
undist_right = cv2.remap(right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

gray_left = cv2.cvtColor(undist_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(undist_right, cv2.COLOR_BGR2GRAY)

stereo_bm = cv2.StereoBM_create(numDisparities=112 - 16, blockSize=19)
disparity_bm = stereo_bm.compute(gray_left, gray_right).astype(np.float32) / 16.0

stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=16,
    numDisparities=112 - 16,
    blockSize=19,
    disp12MaxDiff=3,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2
)
disparity_sgbm = stereo_sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0

disp_bm_norm = cv2.normalize(disparity_bm, None, 0, 255, cv2.NORM_MINMAX)
disp_sgbm_norm = cv2.normalize(disparity_sgbm, None, 0, 255, cv2.NORM_MINMAX)

disp_bm_norm = np.uint8(disp_bm_norm)
disp_sgbm_norm = np.uint8(disp_sgbm_norm)

heatmap_bm = cv2.applyColorMap(disp_bm_norm, cv2.COLORMAP_HOT)
heatmap_sgbm = cv2.applyColorMap(disp_sgbm_norm, cv2.COLORMAP_HOT)

plt.figure(figsize=(15, 15))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
plt.title("lewy")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
plt.title("prawy")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(heatmap_bm)
plt.title("bm")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(heatmap_sgbm)
plt.title("sgbm")
plt.axis("off")

plt.show()

