# 🔮 SpatialAudio

**Empowering the visually impaired** through AI-driven object detection, spatial narration, and interactive queries.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-Camera%20Feed-green?style=flat-square&logo=opencv)
![TTS](https://img.shields.io/badge/TTS-pyttsx3%20%2F%20gTTS-orange?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-GPT%20Powered-black?style=flat-square&logo=openai)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🧠 Motivation

Navigating the world without vision can be overwhelming. SpatialAudio bridges that gap using **YOLOv8-powered object detection**, paired with **real-time audio narration** and **interactive Q&A**.

This project combines **Computer Vision**, **Spatial Reasoning**, and **Text-to-Speech** with **LLM-powered natural language interaction** to create an assistive system that:

- Detects key objects from a live camera feed
- Determines their **position** (left, center, right) and **approximate distance**
- Narrates surroundings in simple spoken sentences
- Responds to natural language queries like *"Is there a person near me?"*

---

## 🚀 How It Works

### 1. Real-Time Capture
The webcam captures a live feed and passes each frame for analysis every few seconds.

### 2. Object Detection
`detection.py` uses YOLOv8 to detect objects and extract:
- Class name (e.g., person, chair, car)
- Bounding box coordinates
- Approximate distance (estimated from bounding box width)
- Position in frame (left, center, right)

### 3. Movement Detection
`movement.py` checks for object movement patterns to highlight potential obstacles or dangers in the user's path.

### 4. Narration & Interaction
- Converts detected object data into spoken audio narration via `narration.py`
- Stores detection data in memory for answering **natural language queries** using an LLM
- Responds contextually based on what's currently detected

**Example interactions:**

| User Query | System Response |
|---|---|
| "Is there anyone near me?" | "Yes, one person is 2 meters ahead." |
| "How many chairs did you detect?" | "5 chairs detected. Nearest is on the right, 1.5 meters away." |
| "What's the nearest object?" | "A chair, approximately 1 meter to your left." |
| "Is there a door to my left?" | "No door detected on your left currently." |

---

## 🛠️ Project Structure

```
SpatialAudio/
├── dataset_cam/        # Custom dataset for training/fine-tuning
├── utils/              # Helper functions (distance, position, labels)
│   └── coco.txt        # COCO class labels
├── models/             # Model files
├── yolov8n.pt          # YOLOv8 nano weights
├── main.py             # Entry point — webcam + detection + narration
├── detection.py        # YOLOv8 inference and bounding box analysis
├── narration.py        # Text-to-speech narration of detected objects
├── annotation.py       # Annotate video feed with bounding boxes
├── movement.py         # Object movement detection logic
├── utils.py            # Shared utility functions
└── requirements.txt    # Python dependencies
```

---

## 🔧 Installation

**Requirements:** Python 3.8+, a webcam, and optionally an OpenAI API key for LLM queries.

```bash
# Clone the repo
git clone https://github.com/Arushi-Srivastava-16/SpatialAudio.git
cd SpatialAudio

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 object detection |
| `opencv-python` | Camera feed and frame processing |
| `pyttsx3` / `gTTS` | Text-to-speech narration |
| `openai` | LLM-powered natural language queries |
| `numpy` | Numerical operations |

---

## 📁 Dataset

A **custom dataset** is included in `dataset_cam/` for training and evaluation. You can substitute your own dataset — just ensure it's annotated in YOLOv8-compatible format (YOLO `.txt` labels).

---

## 🔮 Future Ideas

- GPS-integrated navigation assistance
- Smart Glass / Raspberry Pi deployment
- Voice command activation (hands-free mode)
- Obstacle avoidance with haptic feedback
- Multi-language narration support
- Emergency alert system for sudden hazards

---

## 📄 License

MIT © Arushi Srivastava
```
