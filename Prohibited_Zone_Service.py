###### Create by Bui Quang Minh and friends on 28/11/2024 ######


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
import datetime
from ultralytics import YOLO

# def inside(points, centroid):
#     # Check polygon point
#     if len(points) < 3:
#         return False  

#     polygon = Polygon(points)
#     centroid = Point(centroid)
#     return polygon.contains(centroid)
    


class Prohibited():
    def __init__(self, model_file, detect_class=["person"], frame_width=1280, frame_height=720):
        # Parameters
        self.model_file = model_file  
        self.detect_class = detect_class
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model = YOLO(self.model_file)
        self.conf_threshold = 0.5
        self.class_names = self.model.names  
        self.invaded_times = {}
    def inside(self,points, centroid):
    # Check polygon point
        if len(points) < 3:
            return False  

        polygon = Polygon(points)
        centroid = Point(centroid)
        return polygon.contains(centroid)
    def draw_prediction(self, img, class_id, x, y, x_plus_w, y_plus_h, points, track_id=None):
        label = str(self.class_names[class_id])
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Centroid calculate
        centroid = ((x + x_plus_w) // 2, (y + y_plus_h) // 2)
        cv2.circle(img, centroid, 5, ((0, 255, 0)), -1)

        # View TrackID
        if track_id is not None:
            cv2.putText(img, f"ID: {track_id}", (x_plus_w, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Check if centroid in the prohibited zone
        if self.inside(points, centroid):
            # If track_id is not create, add
            if track_id not in self.invaded_times:
                self.invaded_times[track_id] = datetime.datetime.now(datetime.timezone.utc)
                # print(self.invaded_times[track_id])
            else:
                # Check if person in the prohibited zone over 3sec
                time_in_zone = (datetime.datetime.now(datetime.timezone.utc) - self.invaded_times[track_id]).total_seconds()
                # print(time_in_zone)
                if time_in_zone > 2:
                    img = self.warning(img)  # Warning 
        else:
            # If person get out the prohibited zone, delete
            if track_id in self.invaded_times:
                del self.invaded_times[track_id]

        return self.inside(points, centroid)

    def warning(self, img):
        cv2.putText(img, "Warning !!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        return img

    def detect(self, frame, points):
        results = self.model.track(frame, verbose = False, persist=True, tracker="bytetrack.yaml")  
        
        detections = results[0].boxes  

        for detection in detections:
            box = detection.xyxy[0].numpy()  
            class_id = int(detection.cls)  
            confidence = detection.conf.item()  
            track_id = int(detection.id) if detection.id is not None else None 

            if confidence >= self.conf_threshold and self.class_names[class_id] in self.detect_class:
                x, y, x_plus_w, y_plus_h = map(int, box)  
                self.draw_prediction(frame, class_id, x, y, x_plus_w, y_plus_h, points, track_id)

        return frame
    def click_event(self,event, x, y, flags, points):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            
    def prohibited_zone(self,frame, points):
        for point in points:
            frame = cv2.circle(frame, (point[0], point[1]), 5, (0,0,255), -1)

        frame = cv2.polylines(frame, [np.int32(points)], True, (255,0, 0), thickness=2)
        return frame

points = [] 
def process_video(input_path, output_path, model_file):
    # global points
    model = Prohibited(model_file=model_file)
    cap = cv2.VideoCapture(input_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = model.prohibited_zone(frame, points)
            frame = model.detect(frame, points)

            cv2.imshow("Prohibited Zone", frame)
            out.write(frame)
            cv2.setMouseCallback('Prohibited Zone', model.click_event, points)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('d'):
                points.pop()
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = 'input\prohibited_test.mp4'
    output_video = 'results\prohibited_out.mp4'
    model_path = "model\yolov8n_training.pt"

    process_video(input_video, output_video, model_path)