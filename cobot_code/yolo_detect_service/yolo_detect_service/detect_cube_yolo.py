import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO
import cv2
import numpy as np
import threading
import sys

from pathlib import Path


class ColorDetectionNode(Node) :
    def __init__(self, imgshow_flag=False, yolo_path = None):
        super().__init__('color_detection_node')
        self.imgshow_flag = imgshow_flag
        self._package_path = Path(get_package_share_directory('yolo_detect_service'))
        # print(self._package_path)
        self.publisher_ = self.create_publisher(Detection2DArray, 'cube_detections', 10)
        self.timer = self.create_timer(0.1, self.detect)
        self.bridge = CvBridge()


        # Camera set
        self.cap = cv2.VideoCapture(0)
        self.width =640
        self.height =480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.armforcesX = 345
        self.armforcesY = 93
        self.detect_data=[]
        

        # YOLO
        self.model = YOLO(yolo_path, verbose=False)

        self.edge_width, self.edge_height = 100, 100
        self.contourarea_threshold=100

        if self.imgshow_flag:
            _,self.frame=self.cap.read()
            imgshow_thread=threading.Thread(target=self.imgShow)
            imgshow_thread.start()


    def detect(self):
        
        # The detection logic here is simplified for brevity
        color_list= ["blue", "green", "orange", "purple"]

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to read frame from camera')
            return

        if self.imgshow_flag:
            self.frame=frame

        frame = cv2.flip(frame,0)
        frame = cv2.flip(frame,1)
        
        
        # Run YOLOv8 inference on the frame
        results = self.model(frame)


        for box in results[0].boxes:
            center_x, center_y, bb_width, bb_height = map(int,box.xywh[0,:4])

            idx = int(box.cls[0])

            if results[0].boxes.conf[0].float() >= 0.7:

                self.detect_data.append([color_list[idx],center_x, center_y, bb_width,bb_height])



            
        box_class = ""
        i = 0
        if len(self.detect_data) > 0:
            for box in self.detect_data:
                print(i, box[0])
                i += 1

            print(box_class)
            user = int(input(" >>>>>> "))

            user_box = self.detect_data[user]
            
            # 좌상단 좌표 계산
            x1 = int(user_box[1] - self.edge_width // 2)
            y1 = int(user_box[2] - self.edge_height // 2)

            # 우하단 좌표 계산
            x2 = int(user_box[1] + self.edge_width // 2)
            y2 = int(user_box[2] + self.edge_height // 2)


            cropped_image = frame[y1:y2, x1:x2]

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            # 가우시안 블러 필터 적용
            blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)

            # 케니 엣지 검출 적용
            edges = cv2.Canny(blurred, 50, 150)

            # 윤곽선 찾기
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 윤곽선 그리기
            contour_image = cv2.drawContours(cropped_image, contours, -1, (0, 255, 0), 2)
            for contour in contours:
                epsilon = 0.05 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                area = cv2.contourArea(approx)
                if area > self.contourarea_threshold:  # 임계값을 조절하여 작은 노이즈를 제거
                    # 컨투어의 중심 찾기
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    # 중심점, 너비, 높이, 각도 추출
                    (x, y), (box_width, box_height), angle = rect
                    center = (int(x), int(y))
                    bb_width = int(box_width)
                    bb_height = int(box_height)
                    
                    detections = Detection2DArray()
                    detections.detections = [self.create_detection(user_box[0], x, y, box_width, box_height, angle)]
                    self.publisher_.publish(detections)

                    print("---------------")
                    print(detections)
                    print("---------------")

                    self.detect_data.clear()
                    


            cv2.imshow("edges", edges)







    def create_detection(self, color, centerx, centery, width, height, angle):
        detection = Detection2D()
        detection.id=color
        detection.bbox.center.position.x = float(centerx)
        detection.bbox.center.position.y = float(centery)
        detection.bbox.center.theta = angle
        detection.bbox.size_x = float(width)
        detection.bbox.size_y = float(height)

        return detection
    

    def imgShow(self):
        while True:

            frame=self.frame
            for color,cX,cY,_,_ in self.detect_data:
                # 중심점에 점 그리기
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                # 중심점 좌표 출력
                transformX=cX-self.armforcesX
                transformY=self.armforcesY-cY
                cv2.putText(frame, f'{color} ({transformX}, {transformY})', (cX - 50, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            frame=cv2.line(frame,(self.armforcesX,0),(self.armforcesX,self.height),(0,255,0),1)
            frame=cv2.line(frame,(0,self.armforcesY),(self.width,self.armforcesY),(0,255,0),1)
            frame=cv2.circle(frame,(self.armforcesX,self.armforcesY),2,(0,0,255),-1)
            # 결과 출력
            cv2.imshow('Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) > 1 and sys.argv[1] == 'True':
        imgshow_flag = True
    else:
        imgshow_flag = False
    color_detection_node = ColorDetectionNode(imgshow_flag, '/home/yeseo/ws_moveit2/src/yolo_detect_service/yolo_detect_service/best.pt')
    rclpy.spin(color_detection_node)
    color_detection_node.destroy_node()
    rclpy.shutdown()
