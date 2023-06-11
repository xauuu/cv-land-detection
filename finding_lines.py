import numpy as np
import cv2
from PIL import Image
import matplotlib.image as mpimg


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 56
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_inf = None
        self.curvature = None
        self.deviation = None


def warp_image(img, src, dst, size):
    """
        Chức năng thực hiện phép biến đổi hình thái (perspective transform) cho một ảnh đầu vào.
        Input:
            img: ảnh đầu vào
            src: tọa độ 4 điểm trên ảnh gốc
            dst: tọa độ 4 điểm trên ảnh đích
            size: kích thước ảnh đầu ra
        Output:
            warp_img: ảnh đầu ra sau khi thực hiện phép biến đổi hình thái
            M: ma trận biến đổi
            Minv: ma trận biến đổi ngược
    """
    # matrix M, and its inverse Minv.
    # hàm getPerspectiveTransform() trả về ma trận biến đổi
    M = cv2.getPerspectiveTransform(src, dst)
    # hàm getPerspectiveTransform() trả về ma trận biến đổi ngược
    Minv = cv2.getPerspectiveTransform(dst, src)
    # hàm warpPerspective() thực hiện phép biến đổi hình thái
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv


def rad_of_curvature(left_line, right_line):
    """
    Tính toán bán kính cong của các đường lái xe trái và phải dựa trên tọa độ pixel của chúng
    Input:
        left_line: đường đi bên trái
        right_line: đường đi bên phải
    Output:
        left_line.radius_of_curvature: bán kính cong của đường đi bên trái
        right_line.radius_of_curvature: bán kính cong của đường đi bên phải
    """

    ploty = left_line.ally  # hàm ally() trả về tọa độ y của các điểm trên đường đi bên trái
    # hàm allx() trả về tọa độ x của các điểm trên đường đi bên trái và bên phải
    leftx, rightx = left_line.allx, right_line.allx

    leftx = leftx[::-1]  # Đảo ngược để phù hợp với trên xuống dưới theo y
    rightx = rightx[::-1]  # Đảo ngược để phù hợp với trên xuống dưới theo y

    # Define conversions in x and y from pixels space to meters
    # Định nghĩa các đơn vị chuyển đổi từ không gian pixel sang mét
    # độ rộng của làn đường
    width_lanes = abs(right_line.startx - left_line.startx)
    ym_per_pix = 30 / 720  # mét trên mỗi pixel theo trục y (theo chiều dọc)
    # mét trên mỗi pixel theo trục x (theo chiều ngang)
    xm_per_pix = 3.7*(720/1280) / width_lanes

    # Định nghĩa giá trị y mà chúng ta muốn bán kính cong
    # giá trị y tối đa, tương ứng với đáy của hình ảnh
    y_eval = np.max(ploty)

    # Fit đa thức mới vào x, y trong không gian
    # hàm polyfit() trả về các hệ số của đa thức
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    # hàm polyfit() trả về các hệ số của đa thức
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])  # bán kính cong của đường đi bên trái
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])  # bán kính cong của đường đi bên phải
    # radius of curvature result
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad


def smoothing(lines, pre_lines=3):
    """
    Tính trung bình độ cong của các đường lái xe gần đây, giúp giảm nhiễu và làm cho dự báo ổn định hơn.
    Input:
        lines: danh sách các đường lái xe gần đây
        pre_lines: số lượng đường lái xe gần đây
    Output:
        avg_line: đường lái xe trung bình
    """
    # thu thập các đường lái xe và in đường lái xe trung bình
    # hàm squeeze() loại bỏ các chiều không cần thiết của mảng
    lines = np.squeeze(lines)
    avg_line = np.zeros((720))  # tạo mảng 1 chiều có 720 phần tử có giá trị 0

    # hàm enumerate() trả về một đối tượng liệt kê
    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:  # nếu ii = pre_lines thì dừng vòng lặp
            break
        avg_line += line  # cộng các đường lái xe gần đây
    # tính trung bình độ cong của các đường lái xe gần đây
    avg_line = avg_line / pre_lines

    return avg_line


def blind_search(b_img, left_line, right_line):
    """
    blind search - first frame, lost lane lines
    using histogram & sliding window
    """
    """
    Sử dụng để tìm kiếm đường lái trong một hình ảnh, khi không có đường lái nào được xác định trước hoặc các đường lái bị mất.
    Được sử dụng trong trường hợp đầu tiên, khi chúng ta không có thông tin về các đường lái xe.
    Input:
        b_img: ảnh nhị phân
        left_line: làn đường bên trái, lưu trữ thông tin của đường lái xe bên trái
        right_line: làn đường bên phải, lưu trữ thông tin của đường lái xe bên phải
    Output:
        left_line: làn đường bên trái, lưu trữ thông tin của đường lái xe bên trái
        right_line: làn đường bên phải, lưu trữ thông tin của đường lái xe bên phải
    """
    # Lấy lược đồ của nửa dưới của hình ảnh
    histogram = np.sum(b_img[int(b_img.shape[0] / 2):, :], axis=0)

    # Tạo một hình ảnh đầu ra để vẽ và trực quan hóa kết quả
    output = np.dstack((b_img, b_img, b_img)) * 255

    # Tìm điểm cao nhất của nửa trái và nửa phải của lược đồ
    # Đây sẽ là điểm bắt đầu cho các đường lái xe bên trái và bên phải
    midpoint = np.int(histogram.shape[0] / 2)  # điểm giữa của hình ảnh
    # điểm bắt đầu của đường lái xe bên trái
    start_leftX = np.argmax(histogram[:midpoint])
    # điểm bắt đầu của đường lái xe bên phải
    start_rightX = np.argmax(histogram[midpoint:]) + midpoint

    # Chọn số lượng cửa sổ trượt
    num_windows = 9
    # Đặt chiều cao của cửa sổ
    window_height = np.int(b_img.shape[0] / num_windows)

    # Tìm tất cả các điểm ảnh khác không trong hình ảnh
    nonzero = b_img.nonzero()  # hàm nonzero() trả về các chỉ mục của các phần tử khác 0
    nonzeroy = np.array(nonzero[0])  # tọa độ y của các điểm ảnh khác 0
    nonzerox = np.array(nonzero[1])  # tọa độ x của các điểm ảnh khác 0

    # Đặt lại vị trí hiện tại để cập nhật cho mỗi cửa sổ
    current_leftX = start_leftX
    current_rightX = start_rightX

    # Đặt số lượng điểm ảnh tối thiểu được tìm thấy để trung tâm lại cửa sổ
    min_num_pixel = 50

    # Tạo danh sách rỗng để nhận chỉ mục điểm ảnh của làn đường bên trái và bên phải
    win_left_lane = []  # danh sách các điểm ảnh của đường lái xe bên trái
    win_right_lane = []  # danh sách các điểm ảnh của đường lái xe bên phải

    window_margin = left_line.window_margin  # độ lệch của cửa sổ trượt

    # Tạo vòng lặp cho tất cả các cửa sổ
    for window in range(num_windows):
        # Định danh ranh giới cửa sổ theo x và y (và bên phải và bên trái)
        # giới hạn dưới của cửa sổ
        win_y_low = b_img.shape[0] - (window + 1) * window_height
        win_y_high = b_img.shape[0] - window * \
            window_height    # giới hạn trên của cửa sổ
        win_leftx_min = current_leftX - window_margin  # giới hạn trái của cửa sổ
        win_leftx_max = current_leftX + window_margin   # giới hạn phải của cửa sổ
        win_rightx_min = current_rightX - window_margin  # giới hạn trái của cửa sổ
        win_rightx_max = current_rightX + window_margin  # giới hạn phải của cửa sổ

        # Vẽ các cửa sổ trên hình ảnh trực quan
        cv2.rectangle(output, (win_leftx_min, win_y_low),
                      (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_rightx_min, win_y_low),
                      (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # Định danh các điểm ảnh khác không trong x và y trong cửa sổ
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
            nonzerox <= win_leftx_max)).nonzero()[0]  # chỉ mục của các điểm ảnh khác 0 trong cửa sổ trái
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
            nonzerox <= win_rightx_max)).nonzero()[0]  # chỉ mục của các điểm ảnh khác 0 trong cửa sổ phải
        # Thêm các chỉ mục này vào danh sách
        # thêm chỉ mục của các điểm ảnh khác 0 trong cửa sổ trái vào danh sách
        win_left_lane.append(left_window_inds)
        # thêm chỉ mục của các điểm ảnh khác 0 trong cửa sổ phải vào danh sách
        win_right_lane.append(right_window_inds)

        # Nếu tìm thấy > min_num_pixel điểm ảnh, trung tâm lại cửa sổ tiếp theo trên vị trí trung bình của chúng
        # nếu số lượng điểm ảnh khác 0 trong cửa sổ trái lớn hơn min_num_pixel
        if len(left_window_inds) > min_num_pixel:
            # trung bình các chỉ mục của các điểm ảnh khác 0 trong cửa sổ trái
            current_leftX = np.int(np.mean(nonzerox[left_window_inds]))
        # nếu số lượng điểm ảnh khác 0 trong cửa sổ phải lớn hơn min_num_pixel
        if len(right_window_inds) > min_num_pixel:
            # trung bình các chỉ mục của các điểm ảnh khác 0 trong cửa sổ phải
            current_rightX = np.int(np.mean(nonzerox[right_window_inds]))

    # Nối các mảng chỉ mục
    # nối các chỉ mục của các điểm ảnh khác 0 trong cửa sổ trái
    win_left_lane = np.concatenate(win_left_lane)
    # nối các chỉ mục của các điểm ảnh khác 0 trong cửa sổ phải
    win_right_lane = np.concatenate(win_right_lane)

    # Tạo danh sách các điểm ảnh khác 0 trong cửa sổ trái và phải
    # danh sách các điểm ảnh khác 0 trong cửa sổ trái
    leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
    # danh sách các điểm ảnh khác 0 trong cửa sổ phải
    rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

    # tô màu xanh các điểm ảnh khác 0 trong cửa sổ trái
    output[lefty, leftx] = [255, 0, 0]
    # tô màu đỏ các điểm ảnh khác 0 trong cửa sổ phải
    output[righty, rightx] = [0, 0, 255]

    # Tìm đường cong bậc 2 phù hợp với các điểm ảnh khác 0 trong cửa sổ trái và phải
    # tìm đường cong bậc 2 phù hợp với các điểm ảnh khác 0 trong cửa sổ trái
    left_fit = np.polyfit(lefty, leftx, 2)
    # tìm đường cong bậc 2 phù hợp với các điểm ảnh khác 0 trong cửa sổ phải
    right_fit = np.polyfit(righty, rightx, 2)

    # gán đường cong bậc 2 phù hợp với các điểm ảnh khác 0 trong cửa sổ trái
    left_line.current_fit = left_fit
    # gán đường cong bậc 2 phù hợp với các điểm ảnh khác 0 trong cửa sổ phải
    right_line.current_fit = right_fit

    # Tạo các giá trị x và y để vẽ đường cong bậc 2
    # tạo mảng các giá trị y từ 0 đến chiều cao của ảnh
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    # tạo mảng các giá trị x tương ứng với các giá trị y
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # tạo mảng các giá trị x tương ứng với các giá trị y
    right_plotx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    # thêm các giá trị x tương ứng với các giá trị y vào danh sách
    left_line.prevx.append(left_plotx)
    # thêm các giá trị x tương ứng với các giá trị y vào danh sách
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + \
            left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + \
            right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    left_line.startx, right_line.startx = left_line.allx[len(
        left_line.allx)-1], right_line.allx[len(right_line.allx)-1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    left_line.detected, right_line.detected = True, True
    # print radius of curvature
    rad_of_curvature(left_line, right_line)
    return output


def prev_window_refer(b_img, left_line, right_line):
    """
    refer to previous window info - after detecting lane lines in previous frame
    """
    """
    Hàm này sẽ tìm đường cong bậc 2 phù hợp với các điểm ảnh khác 0 trong cửa sổ trái và phải
    Input:
        b_img: ảnh nhị phân
        left_line: đường cong bậc 2 của cửa sổ trái
        right_line: đường cong bậc 2 của cửa sổ phải
    Output:
        output: ảnh kết quả
    """
    # Create an output image to draw on and  visualize the result
    output = np.dstack((b_img, b_img, b_img)) * 255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set margin of windows
    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + \
        left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + \
        left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + \
        right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + \
        right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # Identify the nonzero pixels in x and y within the window
    left_inds = ((nonzerox >= leftx_min) & (
        nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (
        nonzerox <= rightx_max)).nonzero()[0]

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    output[lefty, leftx] = [255, 0, 0]
    output[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    leftx_avg = np.average(left_plotx)
    rightx_avg = np.average(right_plotx)

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + \
            left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + \
            right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    # goto blind_search if the standard value of lane lines is high.
    standard = np.std(right_line.allx - left_line.allx)

    if (standard > 80):
        left_line.detected = False

    left_line.startx, right_line.startx = left_line.allx[len(
        left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    # print radius of curvature
    rad_of_curvature(left_line, right_line)
    return output


def find_LR_lines(binary_img, left_line, right_line):
    """
    Dùng hàm này để tìm ra 2 đường line trên ảnh binary
    Input là ảnh binary, 2 đường line trước đó
    Output là 2 đường line hiện tại
    """

    # if don't have lane lines info
    if left_line.detected == False:  # nếu chưa có thông tin về 2 đường line
        # thì dùng hàm blind_search để tìm 2 đường line
        return blind_search(binary_img, left_line, right_line)
    # if have lane lines info
    else:
        # nếu có thông tin về 2 đường line thì dùng hàm prev_window_refer để tìm 2 đường line
        return prev_window_refer(binary_img, left_line, right_line)


def draw_lane(img, left_line, right_line, lane_color=(255, 0, 255), road_color=(0, 255, 0)):
    """
    Vẽ 2 đường line lên ảnh gốc
    Input là ảnh gốc, 2 đường line
    Output là ảnh gốc với 2 đường line
    """
    window_img = np.zeros_like(img)  # tạo 1 ảnh có kích thước giống ảnh gốc

    # lấy window_margin của đường line trái
    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx  # lấy allx của 2 đường line
    ploty = left_line.ally  # lấy ally của 2 đường line

    # Tạo 1 polygon để minh họa vùng tìm kiếm
    # Và chuyển đổi các điểm x và y thành định dạng có thể sử dụng cho cv2.fillPoly()
    # tạo 1 array có kích thước (720, 1, 2)
    left_pts_l = np.array(
        [np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack(
        [left_plotx + window_margin/5, ploty])))])  # tạo 1 array có kích thước (720, 1, 2)
    # nối 2 array lại với nhau, tạo 1 array có kích thước (720, 2, 2)
    left_pts = np.hstack((left_pts_l, left_pts_r))
    # tạo 1 array có kích thước (720, 1, 2)
    right_pts_l = np.array(
        [np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
    right_pts_r = np.array([np.flipud(np.transpose(np.vstack(
        [right_plotx + window_margin/5, ploty])))])  # tạo 1 array có kích thước (720, 1, 2)
    # nối 2 array lại với nhau, tạo 1 array có kích thước (720, 2, 2)
    right_pts = np.hstack((right_pts_l, right_pts_r))

    # Vẽ 2 đường line lên ảnh window_img
    # vẽ đường line trái
    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    # vẽ đường line phải
    cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

    # Recast the x and y points into usable format for cv2.fillPoly()
    # tạo 1 array có kích thước (720, 1, 2)
    pts_left = np.array(
        [np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack(
        [right_plotx-window_margin/5, ploty])))])  # tạo 1 array có kích thước (720, 1, 2)
    # nối 2 array lại với nhau, tạo 1 array có kích thước (720, 2, 2)
    pts = np.hstack((pts_left, pts_right))

    # Vẽ 2 đường line lên ảnh window_img
    # vẽ vùng giữa 2 đường line
    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    # kết hợp 2 ảnh lại với nhau
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

    return result, window_img


def road_info(left_line, right_line):
    """
    Tính toán các thông tin về đường đi
    :param left_line: đường line trái
    :param right_line: đường line phải
    :return curvature: độ cong của đường
    :return direction: hướng của đường
    :return road_inf: thông tin về đường
    """
    curvature = (left_line.radius_of_curvature +
                 right_line.radius_of_curvature) / 2  # tính curvature

    direction = ((left_line.endx - left_line.startx) +
                 (right_line.endx - right_line.startx)) / 2  # tính direction

    # nếu curvature > 2000 và direction < 100 thì đường thẳng
    if curvature > 2000 and abs(direction) < 100:
        road_inf = 'No Curve'
        curvature = -1
    elif curvature <= 2000 and direction < - 50:  # nếu curvature <= 2000 và direction < -50 thì đường cua trái
        road_inf = 'Left Curve'
    elif curvature <= 2000 and direction > 50:  # nếu curvature <= 2000 và direction > 50 thì đường cua phải
        road_inf = 'Right Curve'
    else:  # nếu không thì đường thẳng
        if left_line.road_inf != None:
            road_inf = left_line.road_inf
            curvature = left_line.curvature
        else:
            road_inf = 'None'
            curvature = curvature

    center_lane = (right_line.startx + left_line.startx) / \
        2  # tính center_lane
    lane_width = right_line.startx - left_line.startx  # tính lane_width

    center_car = 720 / 2  # tính center_car
    if center_lane > center_car:  # tính deviation
        deviation = 'Left ' + \
            str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
    elif center_lane < center_car:
        deviation = 'Right ' + \
            str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
    else:
        deviation = 'Center'
    left_line.road_inf = road_inf  # lưu thông tin đường
    left_line.curvature = curvature  # lưu curvature
    left_line.deviation = deviation  # lưu deviation

    return road_inf, curvature, deviation


def print_road_status(img, left_line, right_line):
    """
    Hiển thị các thông số của đường lên ảnh
    """
    road_inf, curvature, deviation = road_info(
        left_line, right_line)  # lấy thông tin đường
    cv2.putText(img, 'Road Status', (22, 30), cv2.FONT_HERSHEY_COMPLEX,
                0.7, (80, 80, 80), 2)  # in chữ Road Status lên ảnh

    lane_inf = 'Lane Info : ' + road_inf  # in thông tin đường lên ảnh
    if curvature == -1:  # nếu đường thẳng thì in ra thẳng
        lane_curve = 'Curvature : Straight line'
    else:  # nếu không thì in ra độ cong
        lane_curve = 'Curvature : {0:0.3f}m'.format(
            curvature)  # in thông tin độ cong lên ảnh
    deviate = 'Deviation : ' + deviation  # in thông tin lệch lên ảnh

    cv2.putText(img, lane_inf, (10, 63), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100, 100, 100), 1)
    cv2.putText(img, lane_curve, (10, 83),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(img, deviate, (10, 103), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100, 100, 100), 1)

    return img


def print_road_map(image, left_line, right_line):
    """ print simple road map """
    img = cv2.imread('images/top_view_car.png', -1)
    img = cv2.resize(img, (120, 246))

    rows, cols = image.shape[:2]
    window_img = np.zeros_like(image)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally
    lane_width = right_line.startx - left_line.startx
    lane_center = (right_line.startx + left_line.startx) / 2
    lane_offset = cols / 2 - (2*left_line.startx + lane_width) / 2
    car_offset = int(lane_center - 360)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_pts_l = np.array([np.transpose(np.vstack(
        [right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack(
        [right_plotx + lane_offset - lane_width + window_margin / 4, ploty])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))
    right_pts_l = np.array(
        [np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
    right_pts_r = np.array([np.flipud(np.transpose(
        np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_pts]), (140, 0, 170))
    cv2.fillPoly(window_img, np.int_([right_pts]), (140, 0, 170))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack(
        [right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(
        np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

    #window_img[10:133,300:360] = img
    road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
    window_img = Image.fromarray(window_img)
    img = Image.fromarray(img)
    road_map.paste(window_img, (0, 0))
    road_map.paste(img, (300-car_offset, 590), mask=img)
    road_map = np.array(road_map)
    road_map = cv2.resize(road_map, (95, 95))
    road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)
    return road_map
