import cv2
from ultralytics import YOLO
from utils import calculate_distance, determine_position, load_coco_classes

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")
class_list = load_coco_classes("utils/coco.txt")

def process_frame(frame, confidence_threshold=0.75):
    """Detect objects in the frame and return a list of detections."""
    results = model(frame)
    frame_height, frame_width, _ = frame.shape
    detected_objects = []

    for res in results:
        for bbox in res.boxes:
            if bbox.conf >= confidence_threshold:
                class_id = int(bbox.cls)
                x1, y1, x2, y2 = bbox.xyxy[0].tolist()
                width_pixels = x2 - x1
                area = width_pixels * (y2 - y1)
                distance_cm = calculate_distance(width_pixels)
                position = determine_position((x1, y1, x2, y2), frame_width, frame_height)

                detected_objects.append({
                    'name': model.names[class_id],
                    'confidence': bbox.conf,
                    'bounding_box': (int(x1), int(y1), int(x2), int(y2)),
                    'distance': distance_cm,
                    'class_id': class_id,
                    'position': position,
                    'area': area
                })
    return detected_objects
