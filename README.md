# 🔮SpatialAudio🔮

**Empowering the visually impaired** through AI-driven object detection, spatial narration, and interactive queries.

---

&#x20;

## 🧠 Motivation

Navigating the world without vision can be overwhelming. SpatialAudio aims to bridge that gap using **YOLOv8-powered object detection**, paired with **real-time narration** and **interactive Q\&A support**.

This project combines **Computer Vision**, **Spatial Reasoning**, and **Text-to-Speech** with **LLM-powered natural language interaction** to create an assistive system that:

* Detects key objects from a live camera feed.
* Determines their **position** (left, center, right) and **distance**.
* Narrates surroundings in simple terms to the user.
* Responds to user queries like:

  * "Is there a person near me?"
  * "How many chairs did you detect?"

---

## 🎥 Demo

> *⬇️ Replace this section with actual gifs/screenshots once you run the app.*
> Add images in a `screenshots/` folder and reference them here.

| Detection | Annotated View | Narration |
| --------- | -------------- | --------- |
|           |                |           |

---

## 🛠️ Project Structure

```bash
.
├── dataset_cam/                # Custom dataset used for training/fine-tuning
├── utils/                      # Helper functions (distance calc, position, labels)
│   └── coco.txt                # COCO class labels
├── yolov8n.pt                  # YOLOv8 model weights
├── main.py                     # Main script (runs webcam + detection + narration)
├── detection.py                # YOLOv8 inference and bounding box analysis
├── narration.py                # Audio narration of detected objects
├── annotation.py              # Annotate video feed with bounding boxes
├── movement.py                # Logic to detect object movement
├── requirements.txt           # Python dependencies
└── README.md                  # You're here!
```

---

## 🚀 How It Works

### 1. Real-Time Capture

The webcam captures live feed and passes each frame for analysis every few seconds.

### 2. Object Detection

`detection.py` uses YOLOv8 to detect objects and extract details:

* Class name (e.g., person, chair, car)
* Bounding box
* Distance (approx. using box width)
* Position in frame (left, center, right)

### 3. Movement Check

If enabled, `movement.py` checks for object movement patterns to highlight potential obstacles or dangers.

### 4. Narration & Interaction

* Converts object data into spoken audio narration.
* Stores data in memory for answering **natural language queries** using an LLM API.
* On receiving a query, parses it and responds contextually based on the detected object data.

---

## 🔜 Natural Language Interaction (LLM Support)

Along with audio narration, SpatialAudio enables **interactive dialogue** with the system.

Users can ask:

> 👩‍🌾 "Is there anyone near me?"
> 💺 "How many chairs did you detect?"
> 🔹 "What’s the nearest object?"
> 🚪 "Is there a door to my left?"

The system responds:

> “Yes, one person is 2 meters ahead.”
> “5 chairs detected. Nearest is on the right, 1.5 meters away.”

Uses an **LLM (like GPT)** to understand and answer contextually based on current detection data.

---

## 🔧 Installation

### ✅ Requirements

* Python 3.8+
* OpenCV
* `ultralytics`
* `pyttsx3` or `gTTS` for narration
* Optional: `openai` or other LLM APIs for interaction

### 🔍 Setup

```bash
# Clone the repo
git clone https://github.com/Arushi-Srivastava-16/SpatialAudio.git
cd SpatialAudio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the main app
python main.py
```

---

## 📆 Dataset

We have created and included a **custom dataset** for training and evaluation, located in the `dataset_cam/` directory.

However, you're welcome to use your **own dataset** by updating the paths and structure as needed. Make sure your dataset is annotated in a format compatible with YOLOv8.

---

## 🧠 Future Ideas

* GPS + navigation assistance
* Smart Glass / Raspberry Pi integration
* Voice command system for full hands-free use
* Vibration feedback for silent usage

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙋‍♀️ Author

**Arushi Srivastava**
💼 B.E. Computer Science @ BITS Pilani, Dubai
🔗 [LinkedIn](https://www.linkedin.com/in/arushi-srivastava-49b333243) • [GitHub](https://github.com/Arushi-Srivastava-16)

---

> *"Technology, when inclusive, becomes truly transformative."*
