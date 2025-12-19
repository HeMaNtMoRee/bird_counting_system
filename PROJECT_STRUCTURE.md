# Bird Counting System - Complete Project Structure

## 📦 File Organization

```
bird_counting_system/
│
├── 📄 main.py                          # Core FastAPI application
├── 📄 test_api.py                      # Testing script
├── 📄 requirements.txt                  # Python dependencies
│
├── 📖 README.md                         # Setup & usage guide
├── 📖 IMPLEMENTATION_DETAILS.md        # Technical documentation
├── 📖 PROJECT_STRUCTURE.md             # This file
│
├── 🔧 setup.sh                          # Linux/Mac setup script
├── 🔧 setup.bat                         # Windows setup script
│
├── 📊 sample_output.json                # Example API response
│
├── 📁 output/                           # Generated videos (auto-created)
│   └── annotated_YYYYMMDD_HHMMSS.mp4
│
└── 📁 venv/                             # Virtual environment (created by setup)
    └── ...
```

---

## 📄 File Descriptions

### Core Application Files

#### **main.py** (Primary Application)
- FastAPI server implementation
- YOLOv8 detection pipeline
- ByteTrack-inspired tracking algorithm
- Weight estimation logic
- Video annotation and output generation
- **Lines:** ~400
- **Dependencies:** FastAPI, Ultralytics, OpenCV

**Key Components:**
```python
- BirdTracker class: Custom tracking implementation
- calculate_weight_proxy(): Weight estimation from bbox
- smooth_counts(): Moving average filter
- /health endpoint: Server status
- /analyze_video endpoint: Main processing pipeline
```

#### **test_api.py** (Testing Suite)
- Automated API testing
- Example usage demonstrations
- JSON result validation
- **Lines:** ~150
- **Usage:** `python test_api.py`

---

### Documentation Files

#### **README.md** (User Guide)
- Installation instructions
- API usage examples
- curl command samples
- Troubleshooting guide
- Performance benchmarks
- **Audience:** End users, developers

#### **IMPLEMENTATION_DETAILS.md** (Technical Docs)
- Architecture diagrams
- Algorithm explanations
- Edge case handling
- Optimization strategies
- Accuracy metrics
- **Audience:** Technical reviewers, contributors

#### **PROJECT_STRUCTURE.md** (This File)
- Complete file listing
- File descriptions
- Setup workflows
- **Audience:** New developers, maintainers

---

### Configuration Files

#### **requirements.txt** (Dependencies)
```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
ultralytics==8.1.0
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.1.3
torch==2.1.0
torchvision==0.16.0
```

**Total Size:** ~2GB installed

#### **sample_output.json** (Example Response)
- Reference JSON structure
- Sample time-series data
- Track examples
- Weight estimates
- **Use Case:** API documentation, testing

---

### Setup Scripts

#### **setup.sh** (Linux/Mac Installer)
- Python version check
- Virtual environment creation
- Dependency installation
- Model download
- **Runtime:** 3-5 minutes

**Usage:**
```bash
chmod +x setup.sh
./setup.sh
```

#### **setup.bat** (Windows Installer)
- Same functionality as setup.sh
- Windows-compatible commands
- **Runtime:** 3-5 minutes

**Usage:**
```cmd
setup.bat
```

---

## 🚀 Quick Start Workflow

### First-Time Setup

```bash
# 1. Extract project
unzip bird_counting_system.zip
cd bird_counting_system

# 2. Run setup (Linux/Mac)
./setup.sh

# OR (Windows)
setup.bat

# 3. Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 4. Start server
python main.py
```

### Testing the API

```bash
# Terminal 1: Run server
python main.py

# Terminal 2: Run tests
python test_api.py

# Or use curl
curl -X POST "http://localhost:8000/analyze_video" \
  -F "file=@sample_video.mp4"
```

---

## 📊 Output Files

### Generated During Processing

#### **Annotated Videos**
- **Location:** `output/annotated_YYYYMMDD_HHMMSS.mp4`
- **Content:** 
  - Bounding boxes with track IDs
  - Real-time count dashboard
  - Weight index overlay
- **Codec:** MP4V (H.264 compatible)
- **Size:** ~2-5x original video size

#### **JSON Responses**
- **Format:** Standard JSON
- **Structure:**
  ```json
  {
    "status": "success",
    "video_info": {...},
    "counts": [...],
    "tracks_sample": {...},
    "weight_estimates": {...},
    "artifacts": {...}
  }
  ```
- **Save Location:** Returned via API, can save with `test_api.py`

---

## 🔧 Customization Points

### Modifying Detection Threshold

**File:** `main.py`  
**Line:** ~240

```python
# Change confidence threshold
@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    conf_thresh: float = 0.20,  # Lower = more detections
    iou_thresh: float = 0.7
):
```

### Adjusting Track Memory

**File:** `main.py`  
**Line:** ~60

```python
def __init__(self, max_age=30, min_hits=3):
    # max_age: frames to keep track without detection
    # min_hits: minimum detections before counting
```

### Changing Weight Formula

**File:** `main.py`  
**Line:** ~150

```python
def calculate_weight_proxy(bbox, frame_shape):
    area_pixels = width * height
    weight_index = area_pixels ** 1.5  # Modify exponent here
    return weight_index
```

### Count Smoothing Window

**File:** `main.py`  
**Line:** ~180

```python
def smooth_counts(counts, window_size=5):
    # window_size: larger = smoother, more lag
```

---

## 📦 Submission Package

### ZIP Contents Checklist

```
✓ main.py
✓ test_api.py
✓ requirements.txt
✓ README.md
✓ IMPLEMENTATION_DETAILS.md
✓ PROJECT_STRUCTURE.md
✓ setup.sh
✓ setup.bat
✓ sample_output.json
✓ output/ (empty folder)
```

### Optional Additions

```
□ sample_video.mp4           (test video)
□ annotated_demo.mp4         (processed output example)
□ test_result.json           (actual API response)
□ screenshots/               (dashboard images)
```

---

## 🛠️ Development Workflow

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/new-tracking-algo
   ```

2. **Modify main.py**
   - Add new functions/classes
   - Update API endpoints if needed

3. **Test changes**
   ```bash
   python test_api.py
   ```

4. **Update documentation**
   - README.md for user-facing changes
   - IMPLEMENTATION_DETAILS.md for technical details

5. **Submit**
   ```bash
   git commit -m "Add improved tracking algorithm"
   ```

---

## 📈 Extending the System

### 1. Add Real-Time Streaming

Create new file: `stream.py`

```python
from fastapi import WebSocket
import cv2

@app.websocket("/stream")
async def stream_analysis(websocket: WebSocket):
    # Process RTSP stream
    pass
```

### 2. Add Database Storage

Create new file: `database.py`

```python
from sqlalchemy import create_engine
import pandas as pd

def save_results(data):
    df = pd.DataFrame(data['counts'])
    df.to_sql('bird_counts', engine)
```

### 3. Add Web Dashboard

Create new folder: `frontend/`

```bash
frontend/
├── index.html
├── app.js
└── styles.css
```

---

## 🐛 Troubleshooting Guide

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
source venv/bin/activate  # Activate virtual env
pip install -r requirements.txt
```

### Model Download Fails

**Problem:** YOLOv8 download timeout

**Solution:**
```python
# Manual download
from ultralytics import YOLO
YOLO('yolov8n.pt')  # Will retry
```

### Output Video Corrupted

**Problem:** Cannot play generated MP4

**Solution:**
```bash
# Install system codecs
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # Mac
```

---

## 📞 Support Resources

### Documentation
- README.md: User guide
- IMPLEMENTATION_DETAILS.md: Technical deep-dive
- Ultralytics Docs: https://docs.ultralytics.com

### Testing
- test_api.py: Automated tests
- curl examples in README.md
- Interactive docs: http://localhost:8000/docs

### Community
- YOLOv8 GitHub: https://github.com/ultralytics/ultralytics
- FastAPI Docs: https://fastapi.tiangolo.com

---

## 📝 Version History

### v1.0.0 (Current)
- Initial release
- YOLOv8n detection
- ByteTrack-inspired tracking
- Weight proxy estimation
- FastAPI endpoints
- Video annotation

### Planned Features (v1.1.0)
- GPU auto-detection
- Multi-camera support
- Real-time streaming
- Behavior classification

---

**Last Updated:** December 2024  
**Maintainer:** Bird Counting Team  
**License:** MIT (for project code; see model licenses)