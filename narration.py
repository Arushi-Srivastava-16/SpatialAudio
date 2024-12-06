import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 10)  # Slower speech
narrated_objects = set()

def narrate_objects(detected_objects, counts):
    """Narrate detected objects with counts."""
    global narrated_objects
    currently_detected = {obj['name'] for obj in detected_objects}

    # Announce new objects
    for name, data in counts.items():
        if name not in narrated_objects:
            closest = data['closest']
            count = data['count']
            message = f"Detected {count} {name}(s), closest one at {closest:.2f} centimeters."
            print(message)
            engine.say(message)

    narrated_objects.update(currently_detected)
    engine.runAndWait()

def summarize_counts(detected_objects):
    """Summarize counts and find the closest instance for each detected object."""
    counts = {}
    for obj in detected_objects:
        name = obj['name']
        distance = obj['distance']

        if name not in counts:
            counts[name] = {'count': 0, 'closest': float('inf')}
        counts[name]['count'] += 1
        counts[name]['closest'] = min(counts[name]['closest'], distance)
    return counts

