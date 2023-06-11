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
    Để có được ảnh không méo, chúng ta cần ma trận camera và hệ số méo
    Tính toán chúng với 20 hình ảnh bàn cờ 9 * 6

    :return: ma trận camera và hệ số méo
    """
    # Prepare object points
    # Tạo điểm đối tượng
    objp = np.zeros((6 * 9, 3), np.float32)  # 54 điểm đối tượng
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x, y cordinates

    for fname in images:  # Duyệt qua danh sách các hình ảnh

        img = mpimg.imread(fname)  # Đọc ảnh
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám

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
        objpoints, imgpoints, gray.shape[::-1], None, None)  # Ở đây chúng ta không cần biến rvecs và tvecs

    return mtx, dist  # Trả về ma trận camera và hệ số méo


def undistort(img, mtx, dist):
    """
    Loại bỏ méo khỏi ảnh

    :param img: ảnh đầu vào
    :param mtx: ma trận camera
    :param dist: hệ số méo

    :return: ảnh không méo
    """

    # Sử dụng ma trận camera và hệ số méo để loại bỏ méo
    return cv2.undistort(img, mtx, dist, None, mtx)
