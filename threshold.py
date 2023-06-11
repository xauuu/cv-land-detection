import numpy as np
import cv2


def sobel_xy(img, orient='x', thresh=(20, 100)):  # hàm tính sobel theo x hoặc y
    """
    Định nghĩa một hàm áp dụng Sobel x hoặc y.
    Độ dốc theo hướng x nhấn mạnh các cạnh gần với dọc.
    Độ dốc theo hướng y nhấn mạnh các cạnh gần với ngang.
    """
    # Cân bằng histogram
    if orient == 'x':  # nêu là x thì lấy đạo hàm theo x
        abs_sobel = np.absolute(
            cv2.Sobel(img, cv2.CV_64F, 1, 0))  # lấy đạo hàm theo x
    if orient == 'y':  # nếu là y thì lấy đạo hàm theo y
        abs_sobel = np.absolute(
            cv2.Sobel(img, cv2.CV_64F, 0, 1))  # lấy đạo hàm theo y
    # Chuyển về 8 bit
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))  # chuyển về 8 bit
    # tạo một ma trận 0 có kích thước bằng với scaled_sobel
    binary_output = np.zeros_like(scaled_sobel)
    # gán giá trị 255 cho các phần tử thỏa mãn điều kiện 20 <= scaled_sobel <= 100
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 255

    # Trả về kết quả
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):  # hàm tính độ dốc
    """
    Định nghĩa một hàm trả về độ lớn của độ dốc
    cho một kích thước kernel sobel cụ thể và các giá trị ngưỡng
    """
    # Take both Sobel x and y gradients
    # Tính độ dốc theo x và y
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Tính độ lớn của độ dốc
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Chuyển về 8 bit
    scale_factor = np.max(gradmag)/255
    # Chuyển về kiểu uint8
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Tạo một ảnh nhị phân với các điểm thỏa mãn ngưỡng
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) &
                  (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh=(0.7, 1.3)):  # hàm tính hướng của độ dốc
    # tính toán đạo hàm Sobel theo cả hai hướng x và y
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Lấy giá trị tuyệt đối của hướng độ dốc
    # áp dụng ngưỡng và tạo ra một ảnh nhị phân
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    # gán giá trị 255 cho các phần tử thỏa mãn điều kiện 0.7 <= absgraddir <= 1.3
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
    # Return the binary image
    return binary_output.astype(np.uint8)


def ch_thresh(ch, thresh=(80, 255)):
    binary = np.zeros_like(ch)
    binary[(ch > thresh[0]) & (ch <= thresh[1])] = 255
    return binary


def gradient_combine(img, th_x, th_y, th_mag, th_dir):
    """
    Find lane lines with gradient information of Red channel
    """
    rows, cols = img.shape[:2]  # lấy kích thước của ảnh
    R = img[220:rows - 12, 0:cols, 2]  # lấy kênh màu Red của ảnh

    sobelx = sobel_xy(R, 'x', th_x)  # tính độ dốc theo x
    #cv2.imshow('sobel_x', sobelx)
    sobely = sobel_xy(R, 'y', th_y)  # tính độ dốc theo y
    #cv2.imshow('sobel_y', sobely)
    mag_img = mag_thresh(R, 3, th_mag)  # tính độ lớn của độ dốc
    #cv2.imshow('sobel_mag', mag_img)
    dir_img = dir_thresh(R, 15, th_dir)  # tính hướng của độ dốc
    #cv2.imshow('result5', dir_img)

    # combine gradient measurements
    # tạo một ảnh nhị phân có kích thước bằng với dir_img
    gradient_comb = np.zeros_like(dir_img).astype(np.uint8)
    # gán giá trị 255 cho các phần tử thỏa mãn điều kiện ((sobelx > 1) & (mag_img > 1) & (dir_img > 1)) | ((sobelx > 1) & (sobely > 1))
    gradient_comb[((sobelx > 1) & (mag_img > 1) & (dir_img > 1))
                  | ((sobelx > 1) & (sobely > 1))] = 255

    return gradient_comb


def hls_combine(img, th_h, th_l, th_s):
    # convert to hls color space
    # chuyển ảnh sang không gian màu HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    rows, cols = img.shape[:2]  # lấy kích thước của ảnh
    R = img[220:rows - 12, 0:cols, 2]  # lấy kênh màu Red của ảnh
    # áp dụng ngưỡng cho kênh màu Red
    _, R = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
    # cv2.imshow('red!!!',R)
    H = hls[220:rows - 12, 0:cols, 0]  # lấy kênh màu Hue của ảnh
    L = hls[220:rows - 12, 0:cols, 1]  # lấy kênh màu Lightness của ảnh
    S = hls[220:rows - 12, 0:cols, 2]  # lấy kênh màu Saturation của ảnh

    h_img = ch_thresh(H, th_h)  # áp dụng ngưỡng cho kênh màu Hue
    #cv2.imshow('HLS (H) threshold', h_img)
    l_img = ch_thresh(L, th_l)  # áp dụng ngưỡng cho kênh màu Lightness
    #cv2.imshow('HLS (L) threshold', l_img)
    s_img = ch_thresh(S, th_s)  # áp dụng ngưỡng cho kênh màu Saturation
    #cv2.imshow('HLS (S) threshold', s_img)

    # Two cases - lane lines in shadow or not
    # tạo một ảnh nhị phân có kích thước bằng với s_img
    hls_comb = np.zeros_like(s_img).astype(np.uint8)
    # gán giá trị 255 cho các phần tử thỏa mãn điều kiện ((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1)) | (R > 1)
    hls_comb[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (
        h_img > 1) & (l_img > 1))] = 255  # | (R > 1)] = 255
    #hls_comb[((s_img > 1) & (h_img > 1)) | (R > 1)] = 255
    return hls_comb


def comb_result(grad, hls):
    """ give different value to distinguish them """
    result = np.zeros_like(hls).astype(np.uint8)
    #result[((grad > 1) | (hls > 1))] = 255
    result[(grad > 1)] = 100
    result[(hls > 1)] = 255

    return result
