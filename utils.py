detection_colors = [(0, 255, 0)] * 80  # Adjust for number of COCO classes

def calculate_distance(width_pixels, known_width=60, focal_length=360):
    """Calculate distance from the camera using the bounding box width."""
    return (known_width * focal_length) / width_pixels if width_pixels > 0 else float('inf')

def determine_position(bbox, frame_width, frame_height):
    """Determine the object's position in the frame."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    vertical = "upper" if center_y < frame_height / 2 else "lower"
    horizontal = "left" if center_x < frame_width / 3 else "right" if center_x > 2 * frame_width / 3 else "center"
    return f"{vertical} {horizontal}".strip()

def load_coco_classes(file_path):
    """Load COCO class names from a file."""
    with open(file_path, "r") as file:
        return file.read().split("\n")
