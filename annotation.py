import cv2
from utils import detection_colors

def annotate_frame(frame, detected_objects):
    """Annotate the frame with bounding boxes and labels for detected objects."""
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bounding_box']
        class_id = obj['class_id']
        class_name = obj['name']
        distance_cm = obj['distance']
        position = obj['position']
        color = detection_colors[class_id]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        # Add label and distance text
        label = f"{class_name} {position} {distance_cm:.2f}cm"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
