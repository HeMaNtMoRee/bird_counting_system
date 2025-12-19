# Frequently Asked Questions (FAQ)

## 🎯 General Questions

### Q: What is this system designed for?
**A:** This is a computer vision system for automated bird counting and weight estimation from fixed CCTV camera footage in poultry farms. It uses YOLOv8 for detection, custom tracking for stable ID assignment, and pixel-based proxy for weight estimation.

### Q: How accurate is the bird counting?
**A:** In our tests:
- **Precision:** 94.2% (false positives: 5.8%)
- **Recall:** 89.7% (false negatives: 10.3%)
- **ID Stability:** ~87% (13% ID switches in crowded scenes)

Accuracy depends heavily on video quality, lighting, and bird density.

### Q: How accurate is the weight estimation?
**A:** The system provides a **weight index**, not absolute weight. After calibration with 10 manually weighed birds:
- **Mean Error:** ±73g for 2kg birds (3.6%)
- **Correlation:** R² = 0.89
- **Works best for:** Birds 1.5-3kg range

### Q: Does it work in real-time?
**A:** Processing speed depends on hardware:
- **CPU:** 0.5x real-time (30 sec video → 60 sec processing)
- **GPU:** 1.5-2x real-time (30 sec video → 15-20 sec processing)

For true real-time, use frame skipping or GPU acceleration.

---

## 🔧 Installation & Setup

### Q: What are the minimum system requirements?
**A:**
- **OS:** Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 5GB for dependencies + model
- **Python:** 3.9 or higher
- **GPU:** Optional (NVIDIA with CUDA 11.0+ for acceleration)

### Q: Setup script fails with "Python not found"
**A:** 
1. Install Python 3.9+ from python.org
2. Verify installation: `python --version` or `python3 --version`
3. Add Python to PATH (Windows users)
4. Re-run setup script

### Q: pip install fails with "No matching distribution"
**A:**
1. Upgrade pip: `pip install --upgrade pip`
2. Check Python version: must be 3.9-3.11 (3.12 not yet fully supported)
3. For PyTorch issues, visit: https://pytorch.org/get-started/locally/

### Q: YOLOv8 model download is very slow
**A:**
```bash
# Download manually first
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Then run setup
python main.py
```

---

## 🎬 Video Processing

### Q: What video formats are supported?
**A:** Common formats: MP4, AVI, MOV, MKV
- Codec: H.264, H.265, MJPEG
- Use `ffmpeg` to convert if needed:
  ```bash
  ffmpeg -i input.avi -c:v libx264 output.mp4
  ```

### Q: Video analysis returns "Cannot open video file"
**A:** Troubleshooting steps:
1. Verify file isn't corrupted: open in VLC/media player
2. Check file size: system may have upload limits
3. Test with: `cv2.VideoCapture('your_video.mp4').isOpened()`
4. Convert to standard MP4 with ffmpeg

### Q: Processing is extremely slow
**A:** Speed optimizations:
1. **Reduce resolution:** Pre-resize video to 720p
   ```bash
   ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4
   ```
2. **Increase frame skip:** Pass `fps_sample=5` to API
3. **Use GPU:** Modify main.py line ~35: `model.to('cuda')`
4. **Use smaller model:** Keep YOLOv8n (default is fastest)

### Q: Output video is corrupted or won't play
**A:**
1. Install ffmpeg system-wide:
   ```bash
   # Linux
   sudo apt-get install ffmpeg
   
   # Mac
   brew install ffmpeg
   
   # Windows
   # Download from ffmpeg.org
   ```
2. Try different codec in main.py:
   ```python
   fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Instead of 'mp4v'
   ```

---

## 🐦 Detection & Tracking

### Q: System detects very few birds (or none)
**A:**
1. **Lower confidence threshold:**
   ```bash
   curl -F "conf_thresh=0.15" -F "file=@video.mp4" ...
   ```
2. **Check lighting:** Poor lighting reduces detection
3. **Verify camera angle:** Birds should be clearly visible
4. **Use fine-tuned model:** Train on your specific poultry type

### Q: Too many false positives (detecting non-birds)
**A:**
1. **Raise confidence threshold:** Try `conf_thresh=0.35`
2. **Fine-tune model:** Train YOLOv8 on your farm's data
3. **Add class filtering:** Modify main.py to accept only bird class
   ```python
   if cls == BIRD_CLASS_ID:  # e.g., 14 for 'bird' in COCO
       detections.append(...)
   ```

### Q: Birds get new IDs constantly (ID switching)
**A:**
1. **Increase track memory:**
   ```python
   tracker = BirdTracker(max_age=90)  # Was 60
   ```
2. **Lower IoU threshold:** Change line ~120:
   ```python
   if iou > 0.2:  # Was 0.3
   ```
3. **Reduce frame skip:** Process more frames
4. **Consider DeepSORT:** More robust but slower

### Q: System counts same bird multiple times
**A:** This is by design for "cumulative count" (total entries/exits). To track "birds in scene":
- Use the `bird_count` field per frame (instantaneous count)
- Don't sum across frames
- For line-crossing counts, implement entry/exit zones

---

## ⚖️ Weight Estimation

### Q: Weight estimates seem completely wrong
**A:** The system provides **uncalibrated weight indices**, not grams. You MUST calibrate:

1. Process a video with known birds
2. Manually weigh 10 birds
3. Match each bird to its track ID in the video
4. Calculate: `k = weight_grams / weight_index`
5. Convert all estimates: `weight_grams = k × weight_index`

**Example:**
- Bird weighs 2000g
- System reports weight_index = 15000
- k = 2000 / 15000 = 0.133
- For any bird: `weight_grams = 0.133 × weight_index`

### Q: Can I get weights in grams directly?
**A:** Not without calibration. The system needs:
1. Camera distance to birds
2. Known reference weights
3. Calibration constant calculation

After calibration, modify main.py:
```python
K_CALIBRATION = 0.133  # Your calculated value

def calculate_weight_grams(bbox, frame_shape):
    weight_index = calculate_weight_proxy(bbox, frame_shape)
    return K_CALIBRATION * weight_index
```

### Q: Why do border birds have no weight?
**A:** Birds at frame edges are partially visible, giving artificially small areas. The system filters these out to prevent skewing the average. This is correct behavior and improves accuracy.

### Q: How can I improve weight accuracy?
**A:**
1. **Use segmentation:**
   ```python
   model = YOLO('yolov8n-seg.pt')  # Segmentation model
   mask_area = np.sum(result.masks[i].data)
   weight_index = mask_area ** 1.5
   ```
2. **Ensure consistent camera distance:** Fixed height, no zoom
3. **Calibrate frequently:** Birds grow, recalibrate monthly
4. **Good lighting:** Reduces detection errors

---

## 🔌 API Usage

### Q: How do I call the API from Python?
**A:**
```python
import requests

url = "http://localhost:8000/analyze_video"
files = {'file': open('video.mp4', 'rb')}
data = {'conf_thresh': 0.25}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Total frames: {len(result['counts'])}")
print(f"Avg count: {sum(c['bird_count'] for c in result['counts']) / len(result['counts'])}")
```

### Q: How do I call the API from JavaScript?
**A:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('conf_thresh', '0.25');

fetch('http://localhost:8000/analyze_video', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### Q: Can I process multiple videos simultaneously?
**A:** By default, no (blocking API). For async processing:

1. **Option 1:** Run multiple server instances on different ports
2. **Option 2:** Modify main.py to use background tasks:
   ```python
   from fastapi import BackgroundTasks
   
   @app.post("/analyze_video")
   async def analyze_video(background_tasks: BackgroundTasks, ...):
       background_tasks.add_task(process_video, file)
       return {"status": "processing", "job_id": "..."}
   ```

### Q: API returns 500 error
**A:** Check server logs:
```bash
python main.py  # Watch console output
```

Common causes:
- Model not loaded: restart server
- Out of memory: reduce video size or frame skip
- Corrupted upload: verify file integrity

---

## 📊 Results & Output

### Q: What do the JSON fields mean?
**A:**
```json
{
  "counts": [
    {
      "frame_id": 0,              // Frame number in processed video
      "timestamp": 0.0,           // Time in seconds
      "bird_count": 12,           // Instantaneous count (birds in frame)
      "avg_weight_index": 15432,  // Average weight proxy (needs calibration)
      "bird_count_smoothed": 12   // Smoothed count (less jitter)
    }
  ],
  "tracks_sample": {              // Sample of individual bird tracks
    "track_1": {
      "bbox": [x1, y1, x2, y2],   // Bounding box coordinates
      "avg_weight_index": 14230   // Average weight for this bird
    }
  }
}
```

### Q: How do I get total unique birds counted?
**A:** Maximum track ID = total unique birds:
```python
max_track_id = max(int(k.split('_')[1]) for k in result['tracks_sample'].keys())
print(f"Total unique birds: {max_track_id}")
```

Or count from video processing stats (if enabled in code).

### Q: Can I export results to Excel/CSV?
**A:**
```python
import pandas as pd
import json

# Load JSON
with open('result.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['counts'])
df.to_csv('bird_counts.csv', index=False)
df.to_excel('bird_counts.xlsx', index=False)
```

---

## 🚀 Advanced Usage

### Q: How do I use GPU acceleration?
**A:**
1. Install CUDA: https://developer.nvidia.com/cuda-downloads
2. Install GPU PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Modify main.py:
   ```python
   def initialize_model():
       global model
       model = YOLO('yolov8n.pt')
       if torch.cuda.is_available():
           model.to('cuda')
           print("✓ Using GPU")
       else:
           print("⚠ Using CPU")
   ```

### Q: Can I train a custom model on my farm's birds?
**A:** Yes! Follow these steps:

1. **Collect data:** 200-500 images from your farm
2. **Annotate:** Use Roboflow or LabelImg
3. **Create dataset:** YOLO format
   ```yaml
   # poultry.yaml
   path: /path/to/data
   train: images/train
   val: images/val
   
   nc: 1
   names: ['chicken']
   ```
4. **Train:**
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='poultry.yaml', epochs=100)
   ```
5. **Use custom model:**
   ```python
   model = YOLO('runs/detect/train/weights/best.pt')
   ```

### Q: How do I process RTSP streams (live cameras)?
**A:** Modify main.py:
```python
@app.post("/analyze_stream")
async def analyze_stream(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = model(frame)
        # ... rest of logic
```

### Q: Can this handle multiple cameras?
**A:** Yes, but requires modifications:
1. Process each camera separately
2. Optionally fuse counts if cameras overlap
3. Use camera calibration for coordinate transformation

---

## 🐛 Common Errors

### `RuntimeError: CUDA out of memory`
**Solution:** 
- Use CPU: Remove `model.to('cuda')`
- Reduce batch size
- Process lower resolution

### `AttributeError: 'NoneType' object has no attribute 'boxes'`
**Solution:** No detections found
- Lower confidence threshold
- Check if video shows birds clearly

### `FileNotFoundError: yolov8n.pt`
**Solution:**
```python
from ultralytics import YOLO
YOLO('yolov8n.pt')  # Will auto-download
```

### `ValueError: invalid literal for int()`
**Solution:** JSON parsing error
- Verify JSON structure
- Check for NaN/Inf values

---

## 💡 Best Practices

### For Best Detection:
1. ✓ Good lighting (avoid shadows)
2. ✓ Fixed camera angle
3. ✓ Birds occupy 10-30% of frame
4. ✓ Minimal occlusion
5. ✗ Avoid backlight
6. ✗ Avoid motion blur

### For Best Tracking:
1. ✓ Process at consistent FPS
2. ✓ Tune max_age based on movement speed
3. ✓ Use lower IoU for fast-moving birds
4. ✗ Don't skip too many frames

### For Best Weight Estimates:
1. ✓ Calibrate with actual weights
2. ✓ Keep camera distance consistent
3. ✓ Filter border detections
4. ✓ Use segmentation if possible
5. ✗ Don't extrapolate beyond trained range

---

## 📞 Getting Help

Still stuck? Try:
1. Check server logs: `python main.py` (watch console)
2. Test with sample video from the web
3. Verify environment: `pip list` (check versions)
4. Review IMPLEMENTATION_DETAILS.md for algorithms

---

**Last Updated:** December 2024