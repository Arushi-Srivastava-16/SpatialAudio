import time
import pyttsx3

engine = pyttsx3.init()
DANGER_SIZE_THRESHOLD = 300  # Example value for a "too close" object
tracked_objects = {}

def check_movement(detected_objects):
    """Analyze movement direction and warn about dangers."""
    global tracked_objects
    for obj in detected_objects:
        name = obj['name']
        current_area = obj['area']

        # Check if object was previously tracked
        if name in tracked_objects:
            prev_area = tracked_objects[name]['size']
            direction = "away" if current_area < prev_area else "closer"

            # Warn if object is too close
            if current_area > DANGER_SIZE_THRESHOLD:
                warning = f"Danger: {name} seems too close!"
                print(warning)
                engine.say(warning)

        # Update tracking information
        tracked_objects[name] = {'size': current_area, 'last_seen': time.time()}

    # Clean up old objects not seen recently
    current_time = time.time()
    tracked_objects = {
        k: v for k, v in tracked_objects.items() if current_time - v['last_seen'] < 5
    }

    engine.runAndWait()
