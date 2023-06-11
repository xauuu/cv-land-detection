import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class yolo_tf:
    # Đường dẫn đến tệp trọng số của mô hình YOLO.
    weights_file = 'weights/yolo.ckpt'
    alpha = 0.1  # Hệ số học tập.
    threshold = 0.3  # Ngưỡng phát hiện đối tượng.
    iou_threshold = 0.2  # Ngưỡng phát hiện đối tượng.
    w_img = 1280  # Chiều rộng ảnh đầu vào.
    h_img = 720  # Chiều cao ảnh đầu vào.

    # Khởi tạo biến lưu trữ các hộp giới hạn của đối tượng phát hiện được.
    result_box = None
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]  # Khởi tạo danh sách các lớp đối tượng có thể phát hiện được.

    def __init__(self):
        self.build_networks()

    def build_networks(self):  # Hàm xây dựng mạng YOLO.
        print("Building YOLO_small graph...")
        # Khởi tạo placeholder cho đầu vào của mạng YOLO.
        self.x = tf.placeholder('float32', [None, 448, 448, 3])
        # Khởi tạo lớp tích chập đầu tiên với 64 bộ lọc kích thước 7x7 và bước nhảy là 2.
        self.conv_1 = self.conv_layer(1, self.x, 64, 7, 2)
        # Khởi tạo lớp pooling đầu tiên với kích thước cửa sổ là 2x2 và bước nhảy là 2.
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        # Khởi tạo lớp tích chập thứ hai với 192 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
        # Khởi tạo lớp pooling thứ hai với kích thước cửa sổ là 2x2 và bước nhảy là 2.
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
        # Khởi tạo lớp tích chập thứ ba với 128 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
        # Khởi tạo lớp tích chập thứ tư với 256 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
        # Khởi tạo lớp tích chập thứ năm với 256 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
        # Khởi tạo lớp tích chập thứ sáu với 512 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
        # Khởi tạo lớp pooling thứ ba với kích thước cửa sổ là 2x2 và bước nhảy là 2.
        self.pool_9 = self.pooling_layer(9, self.conv_8, 2, 2)
        # Khởi tạo lớp tích chập thứ bảy với 256 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
        # Khởi tạo lớp tích chập thứ tám với 512 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
        # Khởi tạo lớp tích chập thứ chín với 256 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
        # Khởi tạo lớp tích chập thứ mười với 512 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
        # Khởi tạo lớp tích chập thứ mười một với 256 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
        # Khởi tạo lớp tích chập thứ mười hai với 512 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
        # Khởi tạo lớp tích chập thứ mười ba với 256 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
        # Khởi tạo lớp tích chập thứ mười bốn với 512 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
        # Khởi tạo lớp tích chập thứ mười lăm với 512 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
        # Khởi tạo lớp tích chập thứ mười sáu với 1024 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
        # Khởi tạo lớp pooling thứ tư với kích thước cửa sổ là 2x2 và bước nhảy là 2.
        self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)
        # Khởi tạo lớp tích chập thứ mười bảy với 512 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
        # Khởi tạo lớp tích chập thứ mười tám với 1024 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
        # Khởi tạo lớp tích chập thứ mười chín với 512 bộ lọc kích thước 1x1 và bước nhảy là 1.
        self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
        # Khởi tạo lớp tích chập thứ hai mươi với 1024 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)
        # Khởi tạo lớp tích chập thứ hai mươi mốt với 1024 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_25 = self.conv_layer(25, self.conv_24, 1024, 3, 1)
        # Khởi tạo lớp tích chập thứ hai mươi hai với 1024 bộ lọc kích thước 3x3 và bước nhảy là 2.
        self.conv_26 = self.conv_layer(26, self.conv_25, 1024, 3, 2)
        # Khởi tạo lớp tích chập thứ hai mươi ba với 1024 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_27 = self.conv_layer(27, self.conv_26, 1024, 3, 1)
        # Khởi tạo lớp tích chập thứ hai mươi bốn với 1024 bộ lọc kích thước 3x3 và bước nhảy là 1.
        self.conv_28 = self.conv_layer(28, self.conv_27, 1024, 3, 1)
        self.fc_29 = self.fc_layer(
            29, self.conv_28, 512, flat=True, linear=False)  # Khởi tạo lớp fully connected thứ nhất với 512 đơn vị đầu ra.
        self.fc_30 = self.fc_layer(
            30, self.fc_29, 4096, flat=False, linear=False)  # Khởi tạo lớp fully connected thứ hai với 4096 đơn vị đầu ra.
        # skip dropout_31
        self.fc_32 = self.fc_layer(
            32, self.fc_30, 1470, flat=False, linear=True)  # Khởi tạo lớp fully connected thứ ba với 1470 đơn vị đầu ra.
        self.sess = tf.Session()  # Khởi tạo phiên làm việc của tensorflow.
        # Khởi tạo các biến toàn cục.
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()  # Khởi tạo saver để lưu trọng số.
        # Khôi phục trọng số đã lưu.
        self.saver.restore(self.sess, self.weights_file)
        print("Loading complete!")

    # Hàm xây dựng lớp tích chập.
    def conv_layer(self, idx, inputs, filters, size, stride):
        """
        Lớp convolution được sử dụng để tìm ra các đặc trưng (features) trong ảnh đầu vào bằng cách áp dụng các bộ lọc (filters) lên các vùng nhỏ của ảnh. 
        Các bộ lọc này học cách nhận biết các đặc trưng như cạnh, góc, màu sắc, v.v.
        Các lớp convolution giúp giảm số lượng tham số so với việc kết nối mỗi nút trong một lớp với tất cả các nút trong lớp trước đó. 
        Điều này giúp mô hình có khả năng học được các đặc trưng cục bộ trong ảnh mà không tăng quá nhiều chi phí tính toán và bộ nhớ.

        :param idx: Số thứ tự của lớp tích chập.
        :param inputs: Đầu vào của lớp tích chập.
        :param filters: Số lượng bộ lọc.
        :param size: Kích thước của bộ lọc.
        :param stride: Bước nhảy.

        :return: Lớp tích chập.
        """
        channels = inputs.get_shape()[3]  # Lấy số kênh của đầu vào.
        weight = tf.Variable(tf.truncated_normal(
            [size, size, int(channels), filters], stddev=0.1))  # Khởi tạo trọng số của lớp tích chập.
        # Khởi tạo bias của lớp tích chập.
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2  # Kích thước padding.
        pad_mat = np.array([[0, 0], [pad_size, pad_size],
                           [pad_size, pad_size], [0, 0]])  # Tạo ma trận padding.
        inputs_pad = tf.pad(inputs, pad_mat)  # Thực hiện padding cho đầu vào.

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[
                            1, stride, stride, 1], padding='VALID', name=str(idx)+'_conv')  # Thực hiện tích chập.
        conv_biased = tf.add(conv, biases, name=str(idx)+'_conv_biased')
        print('Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' %
              (idx, size, size, stride, filters, int(channels)))  # In thông tin của lớp tích chập.
        # Áp dụng hàm leaky relu.
        return tf.maximum(self.alpha*conv_biased, conv_biased, name=str(idx)+'_leaky_relu')

    # Hàm xây dựng lớp pooling.
    def pooling_layer(self, idx, inputs, size, stride):
        """
        Lớp pooling được sử dụng để giảm kích thước không gian của đặc trưng và giữ lại các thông tin quan trọng nhất. 
        Thông thường, lớp pooling áp dụng một phép lấy mẫu (sub-sampling) trên vùng cục bộ của đặc trưng, 
        ví dụ như lấy giá trị lớn nhất (max pooling) hoặc lấy giá trị trung bình.
        Lớp pooling giúp giảm độ phức tạp tính toán, giảm overfitting và làm tăng tính bất biến đối với dịch chuyển trong ảnh. 
        Nó cũng giúp tăng cường các đặc trưng quan trọng và giảm khả năng bị nhiễu.

        :param idx: Số thứ tự của lớp pooling.
        :param inputs: Đầu vào của lớp pooling.
        :param size: Kích thước của lớp pooling.
        :param stride: Bước nhảy.

        :return: Lớp pooling.
        """
        print('Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' %
              (idx, size, size, stride))  # In thông tin của lớp pooling.
        # Thực hiện pooling.
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(idx)+'_pool')

    # Hàm xây dựng lớp fully connected.
    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        """
        Lớp fully connected kết nối tất cả các đơn vị đầu vào với đơn vị đầu ra, tạo thành một mạng nơ-ron đầy đủ. 
        Các lớp fc thường được sử dụng ở cuối mạng để chuyển đổi đặc trưng đã được trích xuất thành đầu ra mong muốn 
        (ví dụ: xác suất của các lớp đối tượng trong bài toán nhận diện đối tượng).
        Các lớp fc có khả năng học các mối quan hệ phức tạp giữa các đặc trưng và thực hiện các phép tính phi tuyến. 
        Tuy nhiên, lớp fc cần một số lượng lớn tham số và có khả năng gây overfitting khi dữ liệu huấn luyện không đủ lớn.

        :param idx: Số thứ tự của lớp fully connected.
        :param inputs: Đầu vào của lớp fully connected.
        :param hiddens: Số lượng nơ-ron ẩn.
        :param flat: True nếu đầu vào là một vector.
        :param linear: True nếu không sử dụng hàm kích hoạt.

        :return: Lớp fully connected.
        """
        input_shape = inputs.get_shape().as_list()  # Lấy kích thước của đầu vào.
        if flat:  # Nếu đầu vào là một vector.
            # Tính số chiều của đầu vào.
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            # Chuyển vị ma trận đầu vào.
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            # Thay đổi kích thước của đầu vào.
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:  # Nếu đầu vào không phải là một vector.
            dim = input_shape[1]  # Tính số chiều của đầu vào.
            inputs_processed = inputs  # Không thay đổi kích thước của đầu vào.
        # Khởi tạo ma trận trọng số.
        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        # Khởi tạo vector bias.
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        print('Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (
            idx, hiddens, int(dim), int(flat), 1-int(linear)))  # In thông tin của lớp fully connected.
        if linear:  # Nếu không sử dụng hàm kích hoạt.
            # Thực hiện phép nhân ma trận.
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx)+'_fc')
        # Thực hiện phép nhân ma trận.
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        # Thực hiện hàm kích hoạt ReLU.
        return tf.maximum(self.alpha*ip, ip, name=str(idx)+'_fc')


def detect_from_cvmat(yolo, img):  # Hàm nhận diện đối tượng từ ảnh.
    """
    Hàm nhận diện đối tượng từ ảnh.

    :param yolo: Mạng YOLO.
    :param img: Ảnh đầu vào.

    :return: Ảnh đầu ra.
    """
    s = time.time()  # Lấy thời gian bắt đầu.
    yolo.h_img, yolo.w_img, _ = img.shape  # Lấy kích thước của ảnh.
    img_resized = cv2.resize(img, (448, 448))  # Thay đổi kích thước của ảnh.
    # Chuyển đổi không gian màu của ảnh.
    img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized_np = np.asarray(img_RGB)  # Chuyển ảnh sang dạng numpy array.
    # Khởi tạo mảng đầu vào.
    inputs = np.zeros((1, 448, 448, 3), dtype='float32')
    inputs[0] = (img_resized_np/255.0)*2.0-1.0  # Chuẩn hóa giá trị của ảnh.
    in_dict = {yolo.x: inputs}  # Tạo dictionary đầu vào.
    # Thực hiện nhận diện đối tượng.
    net_output = yolo.sess.run(yolo.fc_32, feed_dict=in_dict)
    # Phân tích kết quả nhận diện.
    result = interpret_output(yolo, net_output[0])
    yolo.result_box = result  # Lưu kết quả nhận diện.
    strtime = str(time.time()-s)  # Tính thời gian thực hiện.
    # In thời gian thực hiện.
    print('Elapsed time : ' + strtime + ' secs' + '\n')


def detect_from_file(yolo, filename):  # Hàm nhận diện đối tượng từ file.
    """
    Hàm nhận diện đối tượng từ file.

    :param yolo: Mạng YOLO.
    :param filename: Đường dẫn đến file.

    :return: Ảnh đầu ra.
    """
    detect_from_cvmat(yolo, filename)  # Thực hiện nhận diện đối tượng từ ảnh.


def interpret_output(yolo, output):
    """
    Hàm phân tích kết quả nhận diện.

    :param yolo: Mạng YOLO.
    :param output: Kết quả nhận diện.

    :return: Kết quả phân tích.
    """
    probs = np.zeros((7, 7, 2, 20))  # Khởi tạo mảng xác suất.
    # Lấy xác suất của các lớp.
    class_probs = np.reshape(output[0:980], (7, 7, 20))
    scales = np.reshape(output[980:1078], (7, 7, 2))  # Lấy tỉ lệ của các hộp.
    # Lấy thông tin của các hộp.
    boxes = np.reshape(output[1078:], (7, 7, 2, 4))
    offset = np.transpose(np.reshape(
        np.array([np.arange(7)]*14), (2, 7, 7)), (1, 2, 0))  # Tạo ma trận offset.

    boxes[:, :, :, 0] += offset  # Cập nhật tọa độ x của các hộp.
    # Cập nhật tọa độ y của các hộp.
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    # Chuẩn hóa tọa độ của các hộp.
    boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
    # Tính diện tích của các hộp.
    boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
    # Tính diện tích của các hộp.
    boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

    boxes[:, :, :, 0] *= yolo.w_img  # Chuẩn hóa tọa độ x của các hộp.
    boxes[:, :, :, 1] *= yolo.h_img  # Chuẩn hóa tọa độ y của các hộp.
    boxes[:, :, :, 2] *= yolo.w_img  # Chuẩn hóa tọa độ x của các hộp.
    boxes[:, :, :, 3] *= yolo.h_img  # Chuẩn hóa tọa độ y của các hộp.

    for i in range(2):  # Duyệt qua 2 hộp.
        for j in range(20):  # Duyệt qua 20 lớp.
            probs[:, :, i, j] = np.multiply(
                class_probs[:, :, j], scales[:, :, i])  # Tính xác suất của các lớp.

    # Lọc các xác suất thấp.
    filter_mat_probs = np.array(probs >= yolo.threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)  # Lọc các hộp thấp.
    boxes_filtered = boxes[filter_mat_boxes[0],
                           filter_mat_boxes[1], filter_mat_boxes[2]]  # Lọc các hộp thấp.
    probs_filtered = probs[filter_mat_probs]  # Lọc các xác suất thấp.
    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
        filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]  # Lọc các lớp thấp.

    # Sắp xếp các xác suất.
    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]  # Sắp xếp các hộp.
    probs_filtered = probs_filtered[argsort]  # Sắp xếp các xác suất.
    classes_num_filtered = classes_num_filtered[argsort]  # Sắp xếp các lớp.

    for i in range(len(boxes_filtered)):  # Duyệt qua các hộp.
        if probs_filtered[i] == 0:  # Nếu xác suất bằng 0 thì bỏ qua.
            continue  # Bỏ qua.
        for j in range(i+1, len(boxes_filtered)):  # Duyệt qua các hộp còn lại.
            # Nếu tỉ lệ giao nhau lớn hơn ngưỡng thì bỏ qua.
            if iou(boxes_filtered[i], boxes_filtered[j]) > yolo.iou_threshold:
                probs_filtered[j] = 0.0  # Gán xác suất bằng 0.

    # Lọc các xác suất thấp.
    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]  # Lọc các hộp thấp.
    probs_filtered = probs_filtered[filter_iou]  # Lọc các xác suất thấp.
    # Lọc các lớp thấp.
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []  # Khởi tạo kết quả.
    for i in range(len(boxes_filtered)):  # Duyệt qua các hộp.
        result.append([yolo.classes[classes_num_filtered[i]], boxes_filtered[i][0],
                      boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])  # Thêm kết quả vào danh sách kết quả.

    return result


def show_results(img, yolo):
    """
    Hiển thị kết quả.

    :params img: Ảnh đầu vào.
    :params yolo: Đối tượng YOLO.

    :returns: Ảnh với kết quả.
    """
    img_cp = img.copy()  # Sao chép ảnh.
    results = yolo.result_box  # Lấy kết quả.
    rect_box = np.zeros_like(img_cp)  # Khởi tạo hộp chữ nhật.
    for i in range(len(results)):  # Duyệt qua các kết quả.
        x = int(results[i][1])  # Tọa độ x.
        y = int(results[i][2])  # Tọa độ y.
        w = int(results[i][3])//2  # Chiều rộng.
        h = int(results[i][4])//2  # Chiều cao.

        cv2.rectangle(rect_box, (x - w, y - h),
                      (x + w, y + h), (125, 125, 125), 10)  # Vẽ hộp chữ nhật.
        # Vẽ hộp chữ nhật.
        cv2.rectangle(rect_box, (x-w, y-h), (x+w, y+h), (255, 125, 0), -1)
        cv2.putText(rect_box, results[i][0], (x-w+5, y-h-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Vẽ chữ.
        # Thêm hộp chữ nhật vào ảnh.
        img_cp = cv2.addWeighted(img_cp, 1, rect_box, 0.3, 0)

        # cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
        #cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    return img_cp  # Trả về ảnh.


def iou(box1, box2):
    """
    Tính tỉ lệ giao nhau.

    :params box1: Hộp 1.
    :params box2: Hộp 2.

    :returns: Tỉ lệ giao nhau.
    """
    tb = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
        max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])  # Tính chiều cao.
    lr = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])  # Tính chiều rộng.
    if tb < 0 or lr < 0:  # Nếu chiều cao hoặc chiều rộng nhỏ hơn 0 thì giao nhau bằng 0.
        intersection = 0
    else:  # Nếu không thì tính tỉ lệ giao nhau.
        intersection = tb*lr
    # Trả về tỉ lệ giao nhau.
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)
