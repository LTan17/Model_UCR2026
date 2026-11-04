import cv2
import numpy as np

class LaneKeeping():
    def __init__(self):
        self.HEIGHT_HORIZON = 80
        self.OFFSET_RATIO = 0.65
        self.CHECKPOINT = 115
        self.calc_speed = 0
        self.zero_speed_counter = 0  # Counter for zero speed
        self.speed_counter = 0  # Counter for non zero-speed conditions
    def find_left_right_points(self, image):
        # Chuyển ảnh sang grayscale và chuẩn hóa nhanh hơn
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        height, width = gray.shape
        black_image = np.copy(image)
        
        if np.all(gray[140] == 0):
            self.calc_speed = 0
            return [], [], black_image

        # Tìm biên trái và biên phải
        left_boundary = np.argmax(gray > 0, axis=1).astype(float)
        right_boundary = (width - np.argmax(gray[:, ::-1] > 0, axis=1)).astype(float)

        # Xử lý hàng không có pixel trắng nào (gán NaN cho các giá trị không hợp lệ)
        left_boundary[:self.HEIGHT_HORIZON][left_boundary[:self.HEIGHT_HORIZON] == 0] = np.nan
        right_boundary[:self.HEIGHT_HORIZON][right_boundary[:self.HEIGHT_HORIZON] == width] = np.nan

        # Lọc nhanh bằng NumPy thay vì vòng lặp for
        valid_left = ~np.isnan(left_boundary) & (left_boundary != 0)
        valid_right = ~np.isnan(right_boundary) & (right_boundary != 320)

        left_lane_coords = np.column_stack((left_boundary[valid_left].astype(int), np.where(valid_left)[0]))
        right_lane_coords = np.column_stack((right_boundary[valid_right].astype(int), np.where(valid_right)[0]))

        return left_lane_coords, right_lane_coords, black_image
    
    def find_middle_points_optimized(self, left_points, right_points, image_height, image_width, black_image):
        """
        Tìm tập điểm giữa dựa trên tập điểm biên trái và phải của làn đường.
        - left_points: Mảng tọa độ (x, y) của biên trái.
        - right_points: Mảng tọa độ (x, y) của biên phải.
        - image_height: Chiều cao của ảnh.
        - image_width: Chiều rộng của ảnh.

        Trả về:
        - middle_points: Danh sách tọa độ (x, y) của điểm giữa.
        - black_image: Ảnh với các điểm đã vẽ.
        """

        OFFSET = 180
        OFFSET_RATIO = 0.7  # Tỷ lệ điều chỉnh offset theo chiều cao
        num_points = 75
        middle_points = []
        STRAIGHT_SPEED = 40 # Tốc độ khi xe đi thẳng
        CURVE_SPEED = 40 # Tốc độ khi xe rẽ

        # Trường hợp có cả hai làn
        if len(left_points) < num_points and len(right_points) < num_points:
            # if self.calc_speed <= STRAIGHT_SPEED:
            #     self.calc_speed += 0.1
            # else:
            #     self.calc_speed = STRAIGHT_SPEED
            self.zero_speed_counter += 1
            self.speed_counter = 0  # Reset non-zero counter when speed is zero
            if self.zero_speed_counter >= 3:  # Only set to 0 after 3 consecutive checks
                self.calc_speed = 0
        else:
            self.zero_speed_counter = 0  # Reset zero-speed counter when speed > 0
            self.speed_counter += 1
            if self.speed_counter >= 3:
                self.calc_speed = STRAIGHT_SPEED if len(left_points) >= num_points and len(right_points) >= num_points else CURVE_SPEED

        if len(left_points) >= num_points and len(right_points) >= num_points:
            # if self.calc_speed <= STRAIGHT_SPEED:
            #     self.calc_speed += 0.1
            # else:
            #     self.calc_speed = STRAIGHT_SPEED
            self.zero_speed_counter = 0  # Reset counter when speed is set to non-zero
            self.calc_speed = STRAIGHT_SPEED
            left_points = np.array(left_points)
            right_points = np.array(right_points)

            # Fit đường cong bậc 2
            left_func = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
            right_func = np.polyfit(right_points[:, 1], right_points[:, 0], 2)

            y_values = np.arange(239, self.HEIGHT_HORIZON, -1)
            x_left = np.polyval(left_func, y_values)
            x_right = np.polyval(right_func, y_values)
            middle_x = (x_left + x_right) / 2

            for i in range(len(y_values)):
                y = y_values[i]
                x1, x2, xm = int(x_left[i]), int(x_right[i]), int(middle_x[i])
                cv2.circle(black_image, (x1, y), 5, (255, 0, 0), -1)
                cv2.circle(black_image, (x2, y), 5, (0, 0, 255), -1)
                cv2.circle(black_image, (xm, y), 5, (0, 255, 0), -1)
                middle_points.append((xm, y))

        # Trường hợp chỉ có làn phải
        elif len(left_points) < num_points and len(right_points) >= num_points:
            # if self.calc_speed >= CURVE_SPEED:
            #     self.calc_speed -= 0.5
            # else:
            #     self.calc_speed = CURVE_SPEED
            self.calc_speed = CURVE_SPEED
            right_points = np.array(right_points)
            right_func = np.polyfit(right_points[:, 1], right_points[:, 0], 2)

            y_values = np.arange(239, self.HEIGHT_HORIZON, -1)
            x_right = np.polyval(right_func, y_values)

            for i in range(len(y_values)):
                y = y_values[i]
                x2 = int(x_right[i])
                xm = int(x2 - OFFSET - OFFSET_RATIO * (y - 240 + 1))
                cv2.circle(black_image, (x2, y), 5, (0, 0, 255), -1)
                cv2.circle(black_image, (xm, y), 5, (0, 255, 0), -1)
                middle_points.append((xm, y))

        # Trường hợp chỉ có làn trái
        elif len(left_points) >= num_points and len(right_points) < num_points:
            # if self.calc_speed >= CURVE_SPEED:
            #     self.calc_speed -= 0.5
            # else:
            #     self.calc_speed = CURVE_SPEED
            self.calc_speed = CURVE_SPEED
            left_points = np.array(left_points)
            left_func = np.polyfit(left_points[:, 1], left_points[:, 0], 2)

            y_values = np.arange(239, self.HEIGHT_HORIZON, -1)
            x_left = np.polyval(left_func, y_values)

            for i in range(len(y_values)):
                y = y_values[i]
                x1 = int(x_left[i])
                xm = int(x1 + OFFSET + OFFSET_RATIO * (y - 240 + 1))
                cv2.circle(black_image, (x1, y), 5, (255, 0, 0), -1)
                cv2.circle(black_image, (xm, y), 5, (0, 255, 0), -1)
                middle_points.append((xm, y))
            

        return middle_points, black_image

    def AngCal(self, image):
        left, right, image = self.find_left_right_points(image)
        middle_points, black_image = self.find_middle_points_optimized(left, right, 240, 320, image)
        middle_points = np.array(middle_points)
        chosen_middle = self.CHECKPOINT
        angle = 0  
        
        if len(middle_points)>0:
            ## Only < chosen middle
            if middle_points[0][1] < chosen_middle:
                x = middle_points[0][0]
                y = middle_points[0][1]
                angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi))
                angle = np.int64(angle/2.3)
                # print(x, y)
                cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
            ## Only > chosen middle
            elif middle_points[-1][1] > chosen_middle:
                x = middle_points[-1][0]
                y = middle_points[-1][1]
                angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi))
                # if(y > chosen_middle and y < chosen_middle + 23): # thres 1
                #     # print('Thres 1')
                #     angle = np.int64(angle/2)
                # elif(y >= chosen_middle + 23 and y < chosen_middle + 23*2): # thres 2
                #     # print('Thres 2')
                #     angle = np.int64(angle/2)
                # elif(y >= chosen_middle + 23*2 and y < chosen_middle + 23*2 + 22): # thres 3
                #     # print('Thres 3')
                #     angle = np.int64(angle/2)
                # elif(y >= chosen_middle + 23*2+22 and y < chosen_middle + 23*2 + 22*2): # thres 4
                #     # print('Thres 4')
                #     angle = np.int64(angle/2)
                # # print(x, y)
                cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
            else:
                for i in sorted(middle_points, key=lambda x: x[1]):
                    if i[1] == chosen_middle:
                        x = i[0]
                        y = i[1]
                        angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi)) 
                        angle = np.int64(angle/2.3)
                        cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
                        # print(x, y)
                        break
        
        angle = -angle
        if angle < 0:
            angle = max(-25, angle)
        else:
            angle = min(25, angle)
        # Send Image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 3
        text_color = (0, 0, 255)  # Màu văn bản: Trắng
        text_position = (10, 20)  # Vị trí của văn bản: Góc trái phía trên
        cv2.putText(black_image, str(int(angle)), text_position, font, font_scale, text_color, font_thickness)
        if self.calc_speed == 0:
            return self.calc_speed, 0, black_image, True
        return self.calc_speed, int(angle), black_image, False