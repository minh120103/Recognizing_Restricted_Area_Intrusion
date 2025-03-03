###### Create by Bui Quang Minh and friends on 28/11/2024 ######

from shapely.geometry import Point, LineString
import cv2
import numpy as np
from ultralytics import YOLO


class PeopleCount():
    def __init__(self, model_file, detect_class=["person"], frame_width=1280, frame_height=720):
        self.model_file = model_file
        self.detect_class = detect_class
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model = YOLO(self.model_file)
        self.conf_threshold = 0.5

        self.previous_centroids = {}  
        self.entry_count = 0  
        self.exit_count = 0  

        # Counting Line 
        self.line = None  

    def draw_prediction(self, img, class_id, x, y, x_plus_w, y_plus_h, track_id):
        label = str(self.model.names[class_id])
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        centroid = ((x + x_plus_w) // 2, (y + y_plus_h) // 2)
        cv2.circle(img, centroid, 5, (0, 255, 0), -1)

        if track_id is not None:
            cv2.putText(img, f"ID: {track_id}", (x_plus_w, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if self.line is not None:
                self.count_entry_exit(track_id, centroid)

    def count_entry_exit(self, track_id, centroid):
        point1, point2 = self.line
        line = LineString([point1, point2])  
        current_point = Point(centroid)

        if track_id in self.previous_centroids:
            previous_point = Point(self.previous_centroids[track_id])
            
            if line.crosses(LineString([previous_point, current_point])):
                if previous_point.y < current_point.y:  
                    self.entry_count += 1
                elif previous_point.y > current_point.y: 
                    self.exit_count += 1

        self.previous_centroids[track_id] = centroid

    def detect(self, frame):
        results = self.model.track(frame, verbose=False, persist=True, tracker="bytetrack.yaml")
        detections = results[0].boxes

        for detection in detections:
            box = detection.xyxy[0].numpy()
            class_id = int(detection.cls)
            confidence = detection.conf.item()
            track_id = int(detection.id) if detection.id is not None else None

            if confidence >= self.conf_threshold and self.model.names[class_id] in self.detect_class:
                x, y, x_plus_w, y_plus_h = map(int, box)
                self.draw_prediction(frame, class_id, x, y, x_plus_w, y_plus_h, track_id)

        if self.line is not None:
            cv2.line(frame, self.line[0], self.line[1], (0, 0, 255), 2)

        cv2.putText(frame, f"Customer In: {self.entry_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Customer Out: {self.exit_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame
    def click_event(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            if len(points) == 2:
                self.line = (points[0], points[1])



points = []
def process_video(input_path, output_path, model_file):
    cap = cv2.VideoCapture(input_path)
    detector = PeopleCount(model_file=model_file)


    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", detector.click_event)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.detect(frame)
        cv2.imshow("Frame", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "input\people_counting_test.mp4" 
    output_video = "results\count_out.mp4"  
    model_path = "model\yolov8s_training.pt"  

    process_video(input_video, output_video, model_path)