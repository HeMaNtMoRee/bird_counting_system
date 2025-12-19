from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import json
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import traceback
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    initialize_model()
    yield
    # Clean up on shutdown (optional)
    print("Shutting down...")

app = FastAPI(title="Bird Counting & Weight Estimation API", lifespan=lifespan) 

# Initialize model globally
model = None
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def initialize_model():
    """Load YOLO model on startup"""
    global model
    try:
        # Using YOLOv8n for speed, can upgrade to yolov8m for accuracy
        model = YOLO('yolov8n.pt')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

class BirdTracker:
    """Simple ByteTrack-inspired tracker for bird counting"""
    
    def __init__(self, max_age=30, min_hits=3):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
        
    def update(self, detections):
        """Update tracks with new detections
        detections: list of [x1, y1, x2, y2, conf, class]
        """
        self.frame_count += 1
        
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        if len(detections) == 0:
            return self.tracks
        
        # Simple IoU-based matching
        if len(self.tracks) == 0:
            # Initialize new tracks
            for det in detections:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'conf': det[4],
                    'age': 0,
                    'hits': 1,
                    'class': int(det[5]) if len(det) > 5 else 0
                }
                self.next_id += 1
        else:
            # Match detections to existing tracks
            matched = set()
            for det in detections:
                best_iou = 0.3  # Minimum IoU threshold
                best_track_id = None
                
                for track_id, track in self.tracks.items():
                    if track_id in matched:
                        continue
                    iou = self._calculate_iou(det[:4], track['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
                
                if best_track_id is not None:
                    # Update existing track
                    self.tracks[best_track_id]['bbox'] = det[:4]
                    self.tracks[best_track_id]['conf'] = det[4]
                    self.tracks[best_track_id]['age'] = 0
                    self.tracks[best_track_id]['hits'] += 1
                    matched.add(best_track_id)
                else:
                    # Create new track
                    self.tracks[self.next_id] = {
                        'bbox': det[:4],
                        'conf': det[4],
                        'age': 0,
                        'hits': 1,
                        'class': int(det[5]) if len(det) > 5 else 0
                    }
                    self.next_id += 1
        
        return self.tracks
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def get_active_tracks(self):
        """Get tracks that have enough hits"""
        return {tid: t for tid, t in self.tracks.items() 
                if t['hits'] >= self.min_hits}

def calculate_weight_proxy(bbox, frame_shape):
    """Calculate weight proxy from bounding box area
    
    Returns weight index (needs calibration constant k for grams)
    Formula: weight_proxy = k * (area_pixels)^1.5
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area_pixels = width * height
    
    # Check if bird is on border (partial detection)
    frame_h, frame_w = frame_shape[:2]
    margin = 5
    on_border = (x1 < margin or y1 < margin or 
                 x2 > frame_w - margin or y2 > frame_h - margin)
    
    if on_border:
        return None  # Ignore partial birds
    
    # Relative weight index (isometric growth assumption)
    weight_index = area_pixels ** 1.5
    
    return weight_index

def smooth_counts(counts, window_size=5):
    """Apply moving average to smooth count time series"""
    if len(counts) < window_size:
        return counts
    
    smoothed = []
    for i in range(len(counts)):
        start = max(0, i - window_size // 2)
        end = min(len(counts), i + window_size // 2 + 1)
        smoothed.append(int(np.mean(counts[start:end])))
    
    return smoothed

@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.7,
    fps_sample: Optional[int] = None
):
    """Analyze video for bird counting and weight estimation"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Save uploaded file temporarily
    temp_input = None
    temp_output = None
    
    try:
        # Create temp file for input
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            temp_input = tmp.name
            shutil.copyfileobj(file.file, tmp)
        
        # Open video
        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame skip
        if fps_sample:
            frame_skip = max(1, fps // fps_sample)
        else:
            # Auto: process every 3rd frame if high FPS
            frame_skip = 3 if fps > 30 else 1
        
        # Output video setup
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"annotated_{timestamp_str}.mp4"
        temp_output = OUTPUT_DIR / output_filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps/frame_skip, (width, height))
        
        # Initialize tracker
        tracker = BirdTracker(max_age=60, min_hits=1)
        
        # Data collection
        time_series = []
        weight_data = defaultdict(list)
        raw_counts = []
        
        frame_idx = 0
        processed_frame_idx = 0
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skipping logic
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Run detection
            results = model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
            
            # Extract detections (filter for bird class if trained on COCO)
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    # Accept all detections (assuming fine-tuned model or bird class)
                    detections.append([x1, y1, x2, y2, conf, cls])
            
            # Update tracker
            tracks = tracker.update(detections)
            active_tracks = tracker.get_active_tracks()
            
            # Count birds
            bird_count = len(active_tracks)
            raw_counts.append(bird_count)
            
            # Calculate weights
            frame_weights = []
            for track_id, track in active_tracks.items():
                weight = calculate_weight_proxy(track['bbox'], frame.shape)
                if weight is not None:
                    weight_data[track_id].append(weight)
                    frame_weights.append(weight)
            
            avg_weight = np.mean(frame_weights) if frame_weights else 0
            
            # Record time series
            timestamp_sec = processed_frame_idx / (fps / frame_skip)
            time_series.append({
                "frame_id": processed_frame_idx,
                "timestamp": round(timestamp_sec, 2),
                "bird_count": bird_count,
                "avg_weight_index": round(avg_weight, 2)
            })
            
            # Annotate frame
            for track_id, track in active_tracks.items():
                x1, y1, x2, y2 = [int(v) for v in track['bbox']]
                
                # Color based on track ID
                color = tuple([int(c) for c in np.random.RandomState(track_id).randint(0, 255, 3)])
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID
                label = f"ID:{track_id}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Dashboard overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, f"Count: {bird_count}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Avg Weight Index: {int(avg_weight)}", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            out.write(frame)
            
            frame_idx += 1
            processed_frame_idx += 1
        
        cap.release()
        out.release()
        
        # Smooth counts
        smoothed_counts = smooth_counts(raw_counts)
        for i, entry in enumerate(time_series):
            if i < len(smoothed_counts):
                entry['bird_count_smoothed'] = smoothed_counts[i]
        
        # Prepare tracks sample
        tracks_sample = {}
        for track_id, weights in list(weight_data.items())[:5]:
            if track_id in active_tracks:
                track = active_tracks[track_id]
                tracks_sample[f"track_{track_id}"] = {
                    "bbox": [float(v) for v in track['bbox']],
                    "avg_weight_index": round(np.mean(weights), 2)
                }
        
        # Calculate aggregate weight estimates
        all_weights = [w for weights in weight_data.values() for w in weights]
        
        response = {
            "status": "success",
            "video_info": {
                "resolution": f"{width}x{height}",
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": processed_frame_idx,
                "frame_skip": frame_skip
            },
            "counts": time_series,
            "tracks_sample": tracks_sample,
            "weight_estimates": {
                "unit": "index (requires calibration for grams)",
                "aggregate_mean": round(np.mean(all_weights), 2) if all_weights else 0,
                "aggregate_std": round(np.std(all_weights), 2) if all_weights else 0,
                "calibration_note": "To convert to grams: weight_grams = k * weight_index, where k is calibration constant from manual weighing"
            },
            "artifacts": {
                "annotated_video": str(output_filename)
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup
        if temp_input and Path(temp_input).exists():
            Path(temp_input).unlink()
        if file.file:
            file.file.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)