import numpy as np
import cv2
import matplotlib.image as mpimg
import glob

# Đọc và tạo danh sách các hình ảnh hiệu chuẩn
images = glob.glob('camera_cal/calibration*.jpg')

# Mảng để lưu các điểm đối tượng và điểm ảnh từ tất cả các hình ảnh
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane


def calib():
    """
    To get an undistorted image, we need camera matrix & distortion coefficient
    Calculate them with 9*6 20 chessboard images
    """
    # Prepare object points
    # Tạo điểm đối tượng
    objp = np.zeros((6 * 9, 3), np.float32)  # 6*9 points, 3D
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x, y coordinates

    for fname in images:

        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tìm góc bàn cờ
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # Nếu tìm thấy góc, thêm điểm đối tượng, điểm ảnh
        if ret == True:
            imgpoints.append(corners)  # 2D
            objpoints.append(objp)  # 3D
        else:
            continue
    # Sử dụng các điểm ảnh và điểm đối tượng để tính toán ma trận camera và hệ số méo
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist


def undistort(img, mtx, dist):
    # Sử dụng ma trận camera và hệ số méo để loại bỏ méo
    return cv2.undistort(img, mtx, dist, None, mtx)
