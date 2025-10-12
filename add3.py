import cv2
import time
from gtts import gTTS
from playsound import playsound
import os
from ultralytics import YOLO
import streamlit as st
import openai
import numpy as np
import speech_recognition as sr
import queue
from pydub import AudioSegment
from pydub.playback import play

# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")
# Speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Initialize YOLO models
model_custom = YOLO("models/up_best.pt")
model_pretrained = YOLO("models/yolov8n.pt")

# Queue for buffered narration
narration_queue = queue.Queue()

# Initialize session state for persistent storage
if 'narrated_objects' not in st.session_state:
    st.session_state.narrated_objects = {}
if 'prev_distances' not in st.session_state:
    st.session_state.prev_distances = {}
if 'object_data' not in st.session_state:
    st.session_state.object_data = []
if 'narration_cooldown' not in st.session_state:
    st.session_state.narration_cooldown = {}
if 'object_summary' not in st.session_state:
    st.session_state.object_summary = []
if 'next_object_id' not in st.session_state:
    st.session_state.next_object_id = 0

# Object widths for distance calculation
object_widths = {
    "chair": 130,
    "stairs": 260,
    "railing": 800,
    "person": 110,
    "exitLight": 50,
    "Lift": 500,
    "door": 600,
    "pillar": 80,
}
default_width = 60

# Deduplication Function
def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def deduplicate_detections(detections, threshold=0.5):
    unique = []
    seen = []
    for det in detections:
        _, name, bbox, _ = det
        if not any(iou(bbox, s) > threshold for s in seen):
            unique.append(det)
            seen.append(bbox)
    return unique

# Distance Calculation
def calculate_distance(width_pixels, object_name, focal_len=500):
    real_width = object_widths.get(object_name, default_width)
    return (real_width * focal_len) / width_pixels if width_pixels > 0 else float('inf')

# Function to calculate movement
def calculate_movement(prev_distance, curr_distance):
    return curr_distance - prev_distance

# Function to determine position
def determine_position(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    vertical = "upper" if center_y < frame_height / 2 else "lower"
    horizontal = (
        "left" if center_x < frame_width / 3 else
        "right" if center_x > frame_width / 3 * 2 else
        "center"
    )
    return f"{vertical} {horizontal}".strip()

def get_persistent_object_id(obj_name, bbox):
    """
    Assign a persistent ID to objects based on their position and name.
    """
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    
    # Check if this object is close to any previously tracked object
    for existing_id, (stored_name, stored_pos) in enumerate(st.session_state.narrated_objects.items()):
        if stored_name == obj_name:
            stored_x, stored_y = stored_pos if isinstance(stored_pos, tuple) else (0, 0)
            if abs(center_x - stored_x) < 100 and abs(center_y - stored_y) < 100:
                return existing_id
    
    # New object - assign new ID
    new_id = st.session_state.next_object_id
    st.session_state.next_object_id += 1
    st.session_state.narrated_objects[new_id] = (obj_name, (center_x, center_y))
    return new_id

def update_object_summary(obj_id, obj_name, curr_distance, movement, count, bbox, frame_width, frame_height):
    position = determine_position(bbox, frame_width, frame_height)

    # Check if the object is already in the summary and update
    for obj in st.session_state.object_summary:
        if obj["id"] == obj_id:
            obj.update({
                "near_dis": curr_distance,
                "movement": movement,
                "count": count,
                "position": position
            })
            return

    # If the object is not in the summary, add it
    st.session_state.object_summary.append({
        "name": obj_name,
        "id": obj_id,
        "count": count,
        "near_dis": curr_distance,
        "movement": movement,
        "position": position
    })

def narrate_objects(detected_objects, frame_width, frame_height, cooldown_seconds, danger_threshold):
    """
    Narrate detected objects with distance, position, movement, and warnings.
    """
    current_time = time.time()

    for temp_id, obj_name, bbox, width_pixels in detected_objects:
        # Get persistent object ID
        obj_id = get_persistent_object_id(obj_name, bbox)
        
        # Calculate distance and movement
        curr_distance = calculate_distance(width_pixels, obj_name)
        prev_distance = st.session_state.prev_distances.get(obj_id, float('inf'))
        movement = calculate_movement(prev_distance, curr_distance)
        position = determine_position(bbox, frame_width, frame_height)
        count = len([o for o in detected_objects if o[1] == obj_name])

        # Check cooldown for narration
        if current_time - st.session_state.narration_cooldown.get(obj_name, 0) > cooldown_seconds:
            # Add general narration to queue
            message = f"{obj_name} detected at {int(curr_distance)} centimeters, in the {position}"
            if not narration_queue.full():
                narration_queue.put(message)

            # Add warning if the object is too close
            if curr_distance < danger_threshold:
                warning_message = f"Warning! {obj_name} too close. Move away!"
                if not narration_queue.full():
                    narration_queue.put(warning_message)

            # Narrate count if multiple objects are detected
            if count > 1:
                count_message = f"{count} {obj_name}s detected"
                if not narration_queue.full():
                    narration_queue.put(count_message)

            # Update cooldown
            st.session_state.narration_cooldown[obj_name] = current_time

        # Update tracking data in session state
        st.session_state.prev_distances[obj_id] = curr_distance
        
        # Add to object_data (keep last 50 entries to avoid memory issues)
        st.session_state.object_data.append([obj_id, obj_name, curr_distance, movement, count])
        if len(st.session_state.object_data) > 50:
            st.session_state.object_data = st.session_state.object_data[-50:]
        
        # Update object summary
        update_object_summary(obj_id, obj_name, curr_distance, movement, count, bbox, frame_width, frame_height)

    # Process narration queue AFTER all objects are processed
    while not narration_queue.empty():
        narration_message = narration_queue.get()
        play_fast_audio(narration_message)
    
    return st.session_state.object_summary

def play_fast_audio(narration_message, speed_factor=1.2):
    """
    Play narration with faster speed using gTTS and pydub.
    """
    try:
        tts = gTTS(text=narration_message, lang='en')
        tts.save("temp.mp3")
        audio = AudioSegment.from_file("temp.mp3")
        fast_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed_factor)})
        fast_audio = fast_audio.set_frame_rate(audio.frame_rate)
        play(fast_audio)
        os.remove("temp.mp3")
    except Exception as e:
        print(f"Audio playback error: {e}")

# GPT-3.5 Turbo Interaction
def get_gpt_response(user_query):
    """
    Generate a response using GPT-3.5/4.
    """
    # Convert object_summary to string
    object_summary_str = str(st.session_state.object_summary)
    if len(object_summary_str) > 500:
        object_summary_str = object_summary_str[:500] + "..."

    # Construct the prompt
    prompt = (
        f"You are an AI assistant helping analyze object detection data and respond to user queries. "
        f"Analyze the given data and answer the query appropriately.\n\n"
        f"Here is the data:\n\n"
        f"1. Object Summary:\n"
        f"This is a summary of all objects detected so far, including their ID, name, distance, movement, and count:\n"
        f"{object_summary_str}\n\n"
        f"2. Narrated Objects:\n"
        f"This tracks objects that have already been narrated to the user:\n"
        f"{st.session_state.narrated_objects}\n\n"
        f"3. Previous Distances:\n"
        f"A dictionary storing the last known distances of objects based on their ID:\n"
        f"{st.session_state.prev_distances}\n\n"
        f"4. Object Data:\n"
        f"A detailed list of detected objects, with each entry in the format [ID, Name, Distance, Movement, Count]:\n"
        f"{st.session_state.object_data}\n\n"
        f"The user has provided this query or statement:\n"
        f"'{user_query}'\n\n"
        f"Your task:\n"
        f"1. Use the provided data to generate a concise and accurate response in form of a sentence.\n"
        f"2. If the question cannot be answered with the given data, explain why and provide guidance.\n"
        f"3. Ensure your response is clear, structured, and directly addresses the query."
    )

    # Call the OpenAI API
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant providing concise answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

def process_video_stream(cap, confidence_threshold, cooldown_seconds, danger_threshold, speed_factor, frame_placeholder):
    """
    Process video stream (camera or uploaded video) with object detection.
    """
    success, frame = cap.read()
    if not success:
        return False

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Perform detection and deduplication
    detections = []
    for model in [model_custom, model_pretrained]:
        results = model(frame)
        for box in results[0].boxes:
            if box.conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            detections.append((id(box), model.names[int(box.cls)], bbox, x2 - x1))

    deduplicated = deduplicate_detections(detections)

    # Narrate objects with proper parameters
    narrate_objects(deduplicated, frame_width, frame_height, cooldown_seconds, danger_threshold)

    # Draw Bounding Boxes
    for _, name, bbox, _ in deduplicated:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update the Streamlit frame display
    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
    
    return True

# ==================== STREAMLIT UI ====================

st.title("Spatial Audio")
st.sidebar.title("Settings")

# Sidebar: Parameters
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
cooldown_seconds = st.sidebar.slider("Narration Cooldown (s)", 0, 20, 10, 1)
distance_threshold = st.sidebar.slider("Distance Threshold (cm)", 0, 500, 300, 10)
danger_threshold = st.sidebar.slider("Danger Threshold (cm)", 0, 200, 50, 5)
speed_factor = st.sidebar.slider("Narration Speed (x)", 0.5, 2.0, 1.2, 0.1)

# Create tabs for Camera and Video Upload
tab1, tab2, tab3 = st.tabs(["📷 Camera", "🎥 Video Upload", "🤖 AI Mode"])

# ==================== TAB 1: CAMERA ====================
with tab1:
    st.header("Camera Feed")
    
    col1, col2 = st.columns(2)
    with col1:
        start_camera_btn = st.button("Start Camera", key="start_camera")
    with col2:
        stop_camera_btn = st.button("Stop Camera", key="stop_camera")
    
    camera_frame_placeholder = st.empty()
    
    if start_camera_btn:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Unable to access camera!")
        else:
            st.success("Camera started!")
            
            while cap.isOpened():
                success = process_video_stream(
                    cap, 
                    confidence_threshold, 
                    cooldown_seconds, 
                    danger_threshold, 
                    speed_factor, 
                    camera_frame_placeholder
                )
                
                if not success or stop_camera_btn:
                    cap.release()
                    st.warning("Camera stopped.")
                    break
            
            cap.release()

# ==================== TAB 2: VIDEO UPLOAD ====================
with tab2:
    st.header("Upload and Process Video")
    
    video_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
    
    if video_file is not None:
        st.success(f"Video uploaded: {video_file.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            open_video_btn = st.button("Open Video", key="open_video")
        with col2:
            close_video_btn = st.button("Close Video", key="close_video")
        
        video_frame_placeholder = st.empty()
        
        if open_video_btn:
            # Save uploaded file temporarily
            temp_file = f"temp_video_{int(time.time())}.{video_file.name.split('.')[-1]}"
            with open(temp_file, "wb") as f:
                f.write(video_file.read())
            
            cap = cv2.VideoCapture(temp_file)
            
            if not cap.isOpened():
                st.error("Unable to open video file!")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                st.info("Processing video...")
                
                while cap.isOpened():
                    success = process_video_stream(
                        cap, 
                        confidence_threshold, 
                        cooldown_seconds, 
                        danger_threshold, 
                        speed_factor, 
                        video_frame_placeholder
                    )
                    
                    if not success or close_video_btn:
                        cap.release()
                        st.warning("Video processing stopped.")
                        break
                
                cap.release()
                
                # Clean up temporary file
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
    else:
        st.info("Please upload a video file to begin.")

# ==================== TAB 3: AI MODE ====================
with tab3:
    st.header("AI Assistant")
    
    # Display current tracking stats
    st.metric("Objects Tracked", len(st.session_state.object_summary))
    st.metric("Total Detections", len(st.session_state.object_data))
    
    # Show recent detections
    if st.session_state.object_summary:
        st.subheader("Recently Detected Objects")
        for obj in st.session_state.object_summary[-5:]:
            st.write(f"- **{obj['name']}** at {int(obj['near_dis'])}cm ({obj['position']})")
    
    st.divider()
    
    # Ensure session state initialization for input
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "Text"
    
    # Select Input Mode
    input_mode = st.radio(
        "Select Input Mode",
        ["Text", "Speech"],
        key="input_mode_ai"
    )
    
    st.subheader("Ask a Question")
    
    if input_mode == "Text":
        user_query = st.text_area(
            "Type your query here:",
            st.session_state.user_query,
            height=100,
            key="query_text_area"
        )
        st.session_state.user_query = user_query
    
    elif input_mode == "Speech":
        if st.button("🎤 Start Listening", key="listen_btn"):
            st.info("Listening for your query...")
            with mic as source:
                try:
                    audio = recognizer.listen(source, timeout=5)
                    st.session_state.user_query = recognizer.recognize_google(audio)
                    st.success(f"Recognized: {st.session_state.user_query}")
                except Exception as e:
                    st.error(f"Speech recognition failed: {e}")
                    st.session_state.user_query = ""
    
    # Submit Query
    if st.button("Submit Query", key="submit_query_btn", type="primary"):
        if st.session_state.user_query:
            with st.spinner("Getting AI response..."):
                response = get_gpt_response(st.session_state.user_query)
                st.success(f"**AI Response:** {response}")
                play_fast_audio(response, speed_factor)
        else:
            st.warning("Please enter a query first!")
    
    # Exit AI Mode
    if st.button("Exit AI Mode", key="exit_ai_btn"):
        st.info("Exited AI mode.")
        exit_msg = "Exiting AI mode. Back to tracking."
        play_fast_audio(exit_msg, speed_factor)
        st.session_state.user_query = ""