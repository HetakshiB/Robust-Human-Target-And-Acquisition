Robust Human Target Detection & Acquisition System
A real-time multi-class object detection system built with YOLOv8 and Flask, featuring admin authentication, alert generation, and comprehensive logging for security monitoring and behavioral analysis.

Table of Contents
Features
Project Overview
Technical Architecture
Installation
Configuration
Usage
Project Structure
API Endpoints
Alert System
Database & Logging
Admin Dashboard
Model Details
Troubleshooting
Performance Optimization
Security Considerations
Contributing
License

Features
Core Detection Capabilities
27-Class Object Detection - Multi-class detection using custom fine-tuned YOLOv8 model

Real-Time Processing - Live webcam/video stream detection with minimal latency

Alert Classification - Three alert classes: knife, gun, mask (customizable)

Dynamic Bounding Boxes - Red boxes for alert classes, green for normal detections

Alert & Notification System
Voice Alerts - Automatic text-to-speech (TTS) generation for threat detection

Visual Alerts - Real-time on-screen alert messages during detection

Logging System - Comprehensive logs with timestamps and detection metadata

Alert Thresholds - Configurable confidence thresholds per class

Web Interface & Authentication
Admin Login - Secure Flask-Login based authentication

Live Dashboard - Real-time video feed with detection visualization

User Management - Admin sign-up and access control

Activity Monitoring - Track system usage and detection events

Session Management - Secure session handling with logout functionality

Data Management
Structured Logging - JSON/CSV export of detection events

Event History - Historical data with filters and search

Performance Metrics - Detection accuracy, processing time, alert frequency

📐 Project Overview
This system is designed for security monitoring and behavioral analysis applications:

Target Use Cases: Access control, threat detection, perimeter monitoring, crowd analysis

Detection Output: Bounding boxes, confidence scores, class labels, timestamps

Processing: CPU/GPU support with model optimization for real-time inference

Scalability: Multi-user administration with centralized dashboard

Technical Architecture
text
┌─────────────────────────────────────────────┐
│         Web Browser / Admin UI              │
│          (Login + Dashboard)                │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│           Flask Backend (app.py)            │
│  ┌──────────────────────────────────────┐  │
│  │ Flask-Login (Authentication)         │  │
│  │ Session Management                   │  │
│  └──────────────────────────────────────┘  │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│     Detection Engine (detector.py)          │
│  ┌──────────────────────────────────────┐  │
│  │ YOLOv8 Model (Fine-tuned on Dataset) │  │
│  │ Real-time Frame Processing           │  │
│  │ Alert Logic & Classification         │  │
│  └──────────────────────────────────────┘  │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  Media & Data Layer                        │
│  ├─ Video Stream (OpenCV)                  │
│  ├─ Audio Alerts (pyttsx3/gTTS)           │
│  ├─ Logs (JSON/CSV)                       │
│  └─ Database (SQLite/PostgreSQL)          │
└─────────────────────────────────────────────┘
 Installation
Prerequisites
Python: 3.8 or higher

OS: Windows, macOS, or Linux

Webcam/Video Source: Connected USB camera or video file

GPU (Optional): CUDA-compatible GPU for faster inference

Step 1: Clone Repository
bash
git clone https://github.com/yourusername/human-target-detection.git
cd human-target-detection
Step 2: Create Virtual Environment
bash
# Using venv (recommended)
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Key Dependencies:

text
flask==2.3.0
flask-login==0.6.2
opencv-python==4.8.0
ultralytics==8.0.0
torch==2.0.0
numpy==1.24.0
pyttsx3==2.90  # For voice alerts
Step 4: Download/Verify Model
bash
# The model will auto-download on first run
# Or manually download your fine-tuned model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
Step 5: Create Directories
bash
mkdir -p logs
mkdir -p templates
mkdir -p static
mkdir -p models

 Configuration
1. Update Flask Configuration (app.py)
python
# Security settings
app.secret_key = 'your-secure-key-here'  # Change this!

# Model settings
MODEL_PATH = 'models/best.pt'  # Path to your fine-tuned model
CONFIDENCE_THRESHOLD = 0.5

# Alert classes
ALERT_CLASSES = ['knife', 'gun', 'mask']
ALERT_CONFIDENCE = 0.6

# Logging
LOG_FILE = 'logs/detection_log.json'
2. Add Users (Users Management)
Edit the users dictionary in app.py:

python
users = {
    'admin': {'password': 'admin123'},
    'supervisor1': {'password': 'pass456'},
    'monitor1': {'password': 'pass789'}
}
⚠️ Security Warning: Use proper password hashing in production:

python
from werkzeug.security import generate_password_hash, check_password_hash

# Generate hashed passwords
hashed = generate_password_hash('admin123')

# Verify passwords
check_password_hash(hashed, 'admin123')
3. Configure Video Source
In detector.py, modify the video capture:

python
# Webcam (default)
cap = cv2.VideoCapture(0)

# Video file
cap = cv2.VideoCapture('path/to/video.mp4')

# Network stream (RTSP)
cap = cv2.VideoCapture('rtsp://camera-ip:554/stream')
📖 Usage
Start the Application
bash
python app.py
Output:

text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Access the Web Interface
Open http://localhost:5000 in your browser

Login with credentials:

Username: admin

Password: admin123

View live video feed on the dashboard

Monitor real-time alerts and logs

Keyboard Controls (If running locally)
text
q  - Quit detection
p  - Pause/Resume
s  - Save current frame

📁 Project Structure
text
human-target-detection/
├── app.py                      # Flask backend & routing
├── detector.py                 # YOLOv8 detection engine
├── models/
│   ├── best.pt                # Fine-tuned model (custom dataset)
│   └── yolov8n.pt             # Base nano model
├── templates/
│   ├── login.html             # Admin login page
│   ├── dashboard.html         # Main dashboard
│   └── logs.html              # Detection logs viewer
├── static/
│   ├── css/
│   │   └── style.css          # Dashboard styling
│   └── js/
│       └── script.js          # Frontend logic
├── logs/
│   ├── detection_log.json     # Detection events
│   ├── alerts.log             # Alert records
│   └── system.log             # System messages
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── config.py                  # Configuration constants
└── utils/
    ├── logger.py              # Logging utility
    ├── audio.py               # TTS & sound alerts
    └── database.py            # Database operations
🔌 API Endpoints
Authentication
Method	Endpoint	Description
GET	/	Login page
POST	/	Process login
GET	/logout	Logout user
Video & Detection
Method	Endpoint	Description
GET	/dashboard	Main detection dashboard
GET	/video_feed	Live video stream (MJPEG)
GET	/api/stats	Detection statistics
GET	/api/alerts	Recent alerts list
Logging
Method	Endpoint	Description
GET	/logs	View detection logs
GET	/api/logs/export	Export logs (JSON/CSV)
DELETE	/api/logs/clear	Clear log history
Example API Call
bash
# Fetch recent alerts (requires authentication)
curl -H "Cookie: session=<session_token>" http://localhost:5000/api/alerts

# Export logs as JSON
curl -H "Cookie: session=<session_token>" http://localhost:5000/api/logs/export?format=json
🚨 Alert System
How Alerts Work
Detection Phase: YOLOv8 detects objects in video frame

Classification: Check if detected class is in ALERT_CLASSES

Threshold Check: Confirm confidence > ALERT_CONFIDENCE

Actions:

🔴 Draw red bounding box around threat

📢 Play voice alert message

📝 Log event with timestamp & metadata

📊 Update dashboard in real-time

Alert Classes & Actions
python
ALERT_CLASSES = {
    'knife': {
        'color': (0, 0, 255),  # Red (BGR)
        'message': 'ALERT: Knife detected!',
        'voice': 'A knife has been detected on the premises',
        'severity': 'HIGH'
    },
    'gun': {
        'color': (0, 0, 255),
        'message': 'CRITICAL: Weapon detected!',
        'voice': 'Critical alert: Firearm detected',
        'severity': 'CRITICAL'
    },
    'mask': {
        'color': (0, 0, 255),
        'message': 'Alert: Face mask detected',
        'voice': 'Face covering detected',
        'severity': 'MEDIUM'
    }
}
Customizing Alerts
Edit detector.py:

python
if label.lower() in ALERT_CLASSES:
    # Draw red box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Generate voice alert
    speak_alert(f"Alert: {label} detected!")
    
    # Log event
    log_alert(label, conf, (x1, y1, x2, y2))
💾 Database & Logging
Log Format (JSON)
json
{
    "timestamp": "2025-01-15T14:32:55.123Z",
    "user": "admin",
    "event_type": "alert",
    "class": "knife",
    "confidence": 0.92,
    "location": [150, 200, 300, 400],
    "bbox_area": 30000,
    "frame_size": [1280, 720],
    "video_source": "webcam"
}
View Logs
In Dashboard:

Navigate to /logs

Filter by class, date range, confidence

Search by username or event type

Via CLI:

bash
# View recent logs
tail -f logs/detection_log.json

# Export to CSV
python utils/logger.py --export --format=csv --output=detection_report.csv

# Filter by alert class
grep '"class":"knife"' logs/detection_log.json
Database Schema (SQLite)
sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    class TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    frame_width INTEGER, frame_height INTEGER,
    video_source TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    detection_id INTEGER NOT NULL,
    severity TEXT,
    voice_alert_played BOOLEAN,
    acknowledged_by TEXT,
    acknowledged_at DATETIME,
    FOREIGN KEY(detection_id) REFERENCES detections(id),
    FOREIGN KEY(acknowledged_by) REFERENCES users(id)
);
👨‍💼 Admin Dashboard
Dashboard Features
Live Video Feed

Real-time detection with bounding boxes

Confidence scores overlay

FPS counter

Statistics Panel

Total detections today/week/month

Alert frequency

Top detected classes

Average confidence scores

Alert History

Sortable table of recent alerts

Filters by class, severity, date

Export functionality

User Activity

Login timestamps

Active sessions

Access logs

System Status

Camera status (connected/disconnected)

Processing FPS

Memory usage

Model info (version, size)

Dashboard HTML Structure
xml
<!DOCTYPE html>
<html>
<head>
    <title>Human Target Detection Dashboard</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="logo">🎯 Detection System</div>
        <div class="user-menu">
            <span>Welcome, {{ current_user.id }}</span>
            <a href="{{ url_for('logout') }}" class="btn-logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <!-- Live Feed Section -->
        <div class="video-section">
            <h2>Live Detection Feed</h2>
            <img src="{{ url_for('video_feed') }}" alt="Live Feed" class="video-stream">
            <div class="stats-overlay">
                <p>FPS: <span id="fps">0</span></p>
                <p>Detections: <span id="det-count">0</span></p>
                <p>Alerts: <span id="alert-count">0</span></p>
            </div>
        </div>

        <!-- Alerts Section -->
        <div class="alerts-section">
            <h2>Recent Alerts</h2>
            <table id="alerts-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>Severity</th>
                        <th>Acknowledged</th>
                    </tr>
                </thead>
                <tbody id="alerts-body"></tbody>
            </table>
        </div>

        <!-- Logs Section -->
        <div class="logs-section">
            <h2>Detection Logs</h2>
            <button onclick="exportLogs('json')">Export JSON</button>
            <button onclick="exportLogs('csv')">Export CSV</button>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>
🤖 Model Details
Fine-Tuning Information
Dataset: Custom dataset with 27 classes including:

Person, face, head

Weapons: knife, gun, pistol, rifle

Masks: face mask, medical mask

Body parts: hands, arms, legs

Accessories & contextual objects

Model Specifications:

text
Architecture: YOLOv8 Nano
Backbone: CSPDarknet
Input Size: 640×640
Parameters: ~3.2M
FLOPs: ~8.7B
Inference Time: ~20-30ms (CPU), ~5-10ms (GPU)
Training Configuration:

text
epochs: 100
batch_size: 16
learning_rate: 0.001
optimizer: SGD
augmentation: [Mosaic, RandomAffine, ColorJitter]
Using Custom Model
python
from ultralytics import YOLO

# Load fine-tuned model
model = YOLO('models/best.pt')

# Inference
results = model(frame, conf=0.5)
Model Validation
bash
# Evaluate model on test dataset
python -c "
from ultralytics import YOLO
model = YOLO('models/best.pt')
metrics = model.val(data='dataset.yaml')
print(f'mAP50: {metrics.box.map50}')
print(f'mAP50-95: {metrics.box.map}')
"
🔧 Troubleshooting
Common Issues & Solutions
Issue	Cause	Solution
"Camera not found"	No webcam/video source	Check device connection, update video_source in detector.py
Low FPS (<10)	CPU bottleneck	Use GPU, reduce input size, use faster model (nano/small)
False positives	Low confidence threshold	Increase CONFIDENCE_THRESHOLD to 0.6-0.7
Login fails	Wrong credentials	Verify username/password in users dict
Model too large	Memory issues	Use smaller variant (nano vs medium)
Video feed not streaming	Port blocked	Check firewall, use different port (app.run(port=5001))
Voice alerts not working	pyttsx3 issue	pip install --upgrade pyttsx3, check speaker output
Debug Mode
Enable verbose logging:

python
# In app.py
app.config['DEBUG'] = True

# In detector.py
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f"Detected: {label} @ {conf:.2f}")
Check System Resources
bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU/Memory
ps aux | grep python
top
⚡ Performance Optimization
For Real-Time Processing
Use GPU (Nvidia CUDA)

python
model = YOLO('models/best.pt')
model.to('cuda')  # Move to GPU
Reduce Input Size

python
results = model(frame, imgsz=416)  # Smaller = faster
Lower Confidence Threshold

python
results = model(frame, conf=0.3, iou=0.4)  # Skip low-confidence
Frame Skipping

python
frame_count = 0
skip_frames = 2

while True:
    success, frame = cap.read()
    if frame_count % skip_frames == 0:
        results = model(frame)
    frame_count += 1
Batch Processing

python
# Process multiple frames in batch for higher throughput
frames_batch = [frame1, frame2, frame3]
results = model(frames_batch)
Benchmarking
bash
# Profile detection speed
python -c "
from ultralytics import YOLO
import cv2
import time

model = YOLO('models/best.pt')
cap = cv2.VideoCapture(0)

times = []
for _ in range(100):
    ret, frame = cap.read()
    start = time.time()
    results = model(frame)
    times.append(time.time() - start)

print(f'Average: {sum(times)/len(times)*1000:.2f}ms')
print(f'FPS: {1000/statistics.mean(times):.1f}')
"
🔐 Security Considerations
Authentication & Authorization
✅ Use strong passwords (>12 chars, mixed case, numbers, symbols)

✅ Implement password hashing (werkzeug.security)

✅ Add rate limiting on login attempts

✅ Use secure session cookies

python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/', methods=['POST'])
@limiter.limit("5 per minute")  # Max 5 login attempts per minute
def login():
    # ... authentication logic
Data Protection
🔒 Encrypt sensitive logs

🔒 Use HTTPS in production

🔒 Sanitize user inputs

🔒 Implement CSRF protection

python
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect(app)
Production Deployment
bash
# Use production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Use reverse proxy (Nginx)
# Enable SSL/TLS certificates
# Set secure headers

Contributing
Contributions are welcome! Please follow these guidelines:

Fork the repository

Create a feature branch: git checkout -b feature/amazing-feature

Commit changes: git commit -m 'Add amazing feature'

Push to branch: git push origin feature/amazing-feature

Open a Pull Request

Development Setup
bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 .
black --check .

Acknowledgments
YOLOv8 - Ultralytics

Flask - Web framework

OpenCV - Computer vision

PyTorch - Deep learning framework

References & Resources
Ultralytics YOLO Documentation

Flask Official Documentation

OpenCV Tutorials

YOLOv8 Custom Training

Flask-Login Documentation

🎯 Roadmap
 Multi-camera support

 GPU acceleration optimizations

 Cloud storage integration

 Advanced analytics dashboard

 Mobile app (Flutter)

 Edge deployment (NVIDIA Jetson)

 Kubernetes deployment

 Real-time notification system (Telegram/Email)

 Video recording with alerts

 Advanced tracking (DeepSORT)

Last Updated: December 2025
Status: Active Development
Version: 1.0.0
