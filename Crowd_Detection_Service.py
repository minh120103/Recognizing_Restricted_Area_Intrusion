###### Create by Bui Quang Minh and friends on 28/11/2024 ######


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO


class CrowdDetect:
    def __init__(self, model_file, detect_class=["person"], frame_width=1280, frame_height=720):
        # Parameters
        self.model_file = model_file
        self.detect_class = detect_class
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model = YOLO(self.model_file)
        self.conf_threshold = 0.3
        self.crowd_min_people = 3  # Minimum person to create a crowd
        self.crowd_max_distance = 100  # Maximum space between each person(pixel)
        self.class_names = self.model.names 

    def draw_person(self, img, x, y, x_plus_w, y_plus_h, class_id, track_id=None):
        label = str(self.class_names[class_id])
        color = (0, 255, 0)  
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if track_id is not None:
            cv2.putText(img, f"ID: {track_id}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def draw_crowd(self, img, cluster_boxes):
        for cluster in cluster_boxes:
            if len(cluster) >= self.crowd_min_people:
                x_min = min([box[0] for box in cluster])
                y_min = min([box[1] for box in cluster])
                x_max = max([box[2] for box in cluster])
                y_max = max([box[3] for box in cluster])

                color = (0, 0, 255)  
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(img, f"Crowd detected: {len(cluster)}", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def detect(self, frame):
        results = self.model.track(frame, verbose=False, persist=True, tracker="bytetrack.yaml")
        detections = results[0].boxes  

        centroids = []
        all_boxes = []

        for detection in detections:
            box = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls)
            confidence = detection.conf.item()
            track_id = int(detection.id.item()) if detection.id is not None else None

            if confidence >= self.conf_threshold and self.class_names[class_id] in self.detect_class:
                x, y, x_plus_w, y_plus_h = map(int, box)
                centroid = ((x + x_plus_w) // 2, (y + y_plus_h) // 2)
                centroids.append(centroid)
                all_boxes.append((x, y, x_plus_w, y_plus_h, class_id, track_id))

                self.draw_person(frame, x, y, x_plus_w, y_plus_h, class_id, track_id)

        if len(centroids) < self.crowd_min_people:
            return frame

        dbscan = DBSCAN(eps=self.crowd_max_distance, min_samples=self.crowd_min_people)
        labels = dbscan.fit_predict(centroids)

        cluster_boxes = {}
        for label, box in zip(labels, all_boxes):
            if label == -1:
                continue
            cluster_boxes.setdefault(label, []).append(box[:4])

        self.draw_crowd(frame, cluster_boxes.values())
        return frame


def process_video(input_path, output_path, model_file):
    crowd_detector = CrowdDetect(model_file=model_file)
    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = crowd_detector.detect(frame)
        out.write(frame)

        cv2.imshow('Crowd Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = "input\crowd_detection_test.mp4" 
    output_video = "results\crowd_out.mp4"  
    model_path = "model\yolov8l_training.pt"  

    process_video(input_video, output_video, model_path)
