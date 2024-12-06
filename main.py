import cv2
import time
from detection import process_frame
from narration import narrate_objects, summarize_counts
from movement import check_movement
from annotation import annotate_frame

# Initialize webcam
cap = cv2.VideoCapture(0)

# Parameters
INTERVAL = 3  # Interval for performing object detection
prev = 0
DIRECTION_PROMPT = False  # User input flag to check direction

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection at regular intervals
    if time.time() - prev >= INTERVAL:
        detected_objects = process_frame(frame)

        # Check object movement and potential dangers
        check_movement(detected_objects)

        # Narrate and annotate detected objects
        count= summarize_counts(detected_objects)
        narrate_objects(detected_objects, count)
        annotate_frame(frame, detected_objects)

        # Update the timestamp
        prev = time.time()

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Check for user prompt to get direction feedback
    key = cv2.waitKey(1)
    if key == ord('d'):  # 'd' to get direction feedback
        DIRECTION_PROMPT = True
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
