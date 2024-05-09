import rclpy
from rclpy.node import Node
from mycobot_interfaces.srv import DetectionRQ
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


class ColorDetectionServiceYolo(Node) :
    def __init__(self, imgshow_flag=False, yolo_path = None):
        super().__init__('color_detection_service_yolo')
        self.srv = self.create_service(DetectionRQ, 'color_detection', self.detect_callback)
        print("service create")

        self.imgshow_flag = imgshow_flag
        self._package_path = Path(get_package_share_directory('yolo_detect_service'))
        # print(self._package_path)
        self.bridge = CvBridge()

        # Camera set
        self.cap = cv2.VideoCapture(2)
        self.width =640
        self.height =480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.armforcesX = 288
        self.armforcesY = 288
        self.detect_data=[]
        self.send_data = []


        # YOLO
        self.model = YOLO(yolo_path, verbose=False)

        self.edge_width, self.edge_height = 100, 100
        self.contourarea_threshold=100

        if self.imgshow_flag:
            _,self.frame=self.cap.read()
            imgshow_thread=threading.Thread(target=self.imgShow)
            imgshow_thread.start()


    def detect_callback(self, request:DetectionRQ.Request, response:DetectionRQ.Response):
        # 여기서 색상 감지 로직을 처리하고, 결과를 response에 채워서 반환합니다.
        # request에는 필요한 요청 정보가 포함될 수 있습니다.
        # response에는 처리 결과를 담아서 클라이언트에게 반환합니다.
        if request.trigger:
            response.result = self.detect()  # detect 메서드는 Detection2DArray 메시지를 반환하도록 작성되어야 합니다.

        return response
    

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


        for i, item in enumerate(self.detect_data):

            
            # 좌상단 좌표 계산
            x1 = max(0, int(item[1] - self.edge_width // 2))
            y1 = max(0, int(item[2] - self.edge_height // 2))

            # 우하단 좌표 계산
            x2 = min(self.width - 10, item[1] + self.edge_width // 2) 
            y2 = min(self.height - 10, item[2] + self.edge_height // 2) 



            cropped_image = frame[y1:y2, x1:x2]
            print(cropped_image.shape)
            

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

                    self.send_data.append([item[0],item[1],item[2],bb_width,bb_height,angle])
                    
            

           
        detections = Detection2DArray()
        detections.detections = [self.create_detection(color, centerx, centery, width, height, angle)
                                for color, centerx, centery, width, height, angle in self.send_data]
        self.detect_data.clear()
        self.send_data.clear()
        
        return detections





    # detection 
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

            _,frame=self.cap.read()
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
    
    color_detection_service = ColorDetectionServiceYolo(imgshow_flag, '/home/yeseo/ros2_ws/src/cobot_yolo_pc/cobot_yolo_pc/best.pt')
    rclpy.spin(color_detection_service)
    color_detection_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()








