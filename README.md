# 👁️ VisionAI: OpenCV & MediaPipe Computer Vision Suite

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C7B7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C?style=for-the-badge&logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An intelligent, multi-module Computer Vision & Human-Computer Interaction (HCI) repository. This project features real-time object tracking, person footfall counting with strict margin validation, touchless gesture-controlled calculation, sign language dataset recording, and hand landmark classification.

---

## 🌟 Key Modules & Features

### 1. 🚶 Strict Classroom & Footfall Counter (`counter.py`)
- **Real-Time Tracking**: Leverages YOLOv8 Nano (`yolov8n.pt`) with persistent multi-object tracking for human detection (`class 0`).
- **Interactive Boundary Line**: Click 2 points on the live camera stream to define a bidirectional threshold line.
- **Margin Band Geometry**: Automatically generates virtual safety margins above and below the line with vector normal projections to filter out micro-movements, jitter, and occlusions.
- **Strict Validation**: Requires minimum residence frames (`MIN_FRAMES_IN_MARGIN = 6`) and spatial movement (`MIN_MOVE_PIXELS = 25`) before registering valid **IN** or **OUT** crossings.
- **Noise Rejection**: Filters out small objects, bags, and partial detections using strict bounding box height limits (`MIN_HEIGHT = 130`).

### 2. 🧮 Touchless Gesture Calculator (`gesture_calc.py`)
- **Contactless Arithmetic**: Perform mathematical calculations in mid-air using webcam input and hand tracking.
- **Finger Counting (0–5)**: Extend fingers to input digits `0` to `5`.
- **Intuitive Operators**:
  - 👍 **Thumbs Up**: Addition (`+`)
  - 👎 **Thumbs Down**: Subtraction (`-`)
  - 👌 **OK Gesture** (Thumb & Index pinch): Multiplication (`*`)
  - 🖐 **Hold 5 Fingers**: Compute equals (`=`)
  - ✊ **Hold Fist**: Clear display (`C`)
- **Debounce & Cooldown Engine**: Prevents unintended double-triggers using a tokenized timestamp cooldown and stable-frame thresholding.
- **Real-time HUD**: Displays current mathematical expression, calculated output, detected gesture, and action confirmation.

### 3. 🖐️ Hand Gesture & Finger State Visualizer (`gesture_cam.py`)
- **21-Landmark Hand Tracking**: Real-time 3D landmark detection using MediaPipe Hands.
- **Finger Status Telemetry**: Analyzes knuckle (PIP) vs tip positions to determine whether individual fingers (Thumb, Index, Middle, Ring, Pinky) are extended (`↑`) or folded (`–`).
- **Dynamic Classification**: Identifies gestures including *Open Palm*, *Fist*, *Thumbs Up/Down*, *OK*, and exact finger count.
- **Handedness Detection**: Accurately classifies *Left* vs *Right* hand.
- **Interactive Controls**: Mirror toggle (`m`) and quick exit (`q`).

### 4. 🤟 Sign Language Dataset Recorder (`signs_record.py`)
- **Dual-Hand 3D Feature Extraction**: Employs MediaPipe Holistic to extract 126-dimensional feature vectors (21 landmarks × 3 coordinates × 2 hands) per frame.
- **Pre-Configured Vocabulary**: Captures standard signs:
  `hello`, `thank_you`, `yes`, `no`, `please`, `good`, `morning`, `night`, `help`, `love`.
- **Automated Sequence Pipeline**: Records consistent 50-frame clips (~2s @ 25fps) triggered by the spacebar and exports them as normalized NumPy arrays (`.npy`) along with a `labels.json` metadata file ready for LSTM, GRU, or Transformer training.

### 5. 📑 Presentation Generator (`ppt.py`)
- Programmatically generates professional slide decks (`IoT_Pathshala_Pitch.pptx`) using `python-pptx`, designed for hands-on IoT curriculum, workshop planning, and Atal Tinkering Lab (ATL) pitches.

---

## 📁 Repository Structure

```
.
├── counter.py           # Strict line-crossing person counter using YOLOv8
├── gesture_calc.py      # Air-gesture powered arithmetic calculator
├── gesture_cam.py       # Real-time hand landmark & gesture telemetry visualizer
├── signs_record.py      # Dual-hand sign language dataset recording tool
├── ppt.py               # Slide deck generator for IoT / workshop presentations
├── yolov8n.pt           # YOLOv8 nano pre-trained weights
├── requirements.txt     # Python project dependencies
├── .gitignore           # Git ignore configurations
└── README.md            # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ installed
- Working webcam / camera access
- macOS, Linux, or Windows

### 1. Clone the Repository
```bash
git clone https://github.com/Yuvraj9198/Computer-Vision-Gesture-Suite.git
cd Computer-Vision-Gesture-Suite
```

### 2. Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for macOS Users**: Ensure Terminal or VS Code has been granted Camera permissions in **System Settings > Privacy & Security > Camera**.

---

## 🚀 Running the Modules

### Run the Strict People Counter
```bash
python counter.py
```
- Click **two points** on the video feed to set the reference boundary line.
- The system will draw the reference line and upper/lower margin boundaries.
- As people cross from one side to the other, the `IN` and `OUT` counters will update automatically.
- Press `ESC` to quit.

---

### Run the Gesture Calculator
```bash
python gesture_calc.py
```
| Gesture | Action | Description |
| :--- | :---: | :--- |
| **0 to 5 Fingers** | `0` – `5` | Input digit into current expression |
| **Thumbs Up** 👍 | `+` | Add operator |
| **Thumbs Down** 👎 | `-` | Subtract operator |
| **OK Pinch** 👌 | `*` | Multiply operator |
| **Hold 5 Fingers** (30 frames) | `=` | Calculate & show result |
| **Hold Fist** (30 frames) | `C` | Clear expression & result |
| **Key 'm'** | Mirror | Toggle camera mirror mode |
| **Key 'q'** | Exit | Close calculator |

---

### Run the Hand Gesture Telemetry Demo
```bash
python gesture_cam.py
```
- Displays live hand landmarks, classified gesture name, handedness, and individual finger extension states.
- Press `m` to toggle mirror mode, or `q` to quit.

---

### Record Sign Language Datasets
```bash
python signs_record.py
```
- Cycles through target vocabulary labels.
- Press `SPACE` to record a 50-frame clip.
- Press `n` to skip to the next label.
- Output arrays are saved to `sign_dataset/<label>_<idx>.npy`.

---

## 🛠️ Technology Stack

- **Computer Vision**: [OpenCV (opencv-python)](https://opencv.org/)
- **Machine Learning & Hand Tracking**: [Google MediaPipe](https://developers.google.com/mediapipe)
- **Object Detection & Multi-Object Tracking**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Deep Learning Runtime**: [PyTorch](https://pytorch.org/) & [TorchVision](https://pytorch.org/vision/)
- **Data & Numerical Computing**: [NumPy](https://numpy.org/)
- **Document Automation**: [python-pptx](https://python-pptx.readthedocs.io/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use and adapt for academic, personal, or commercial projects.
