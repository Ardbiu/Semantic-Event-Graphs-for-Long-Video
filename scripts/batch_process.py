import glob
import os
import torch
import sys
import yaml
import argparse
import time
from pathlib import Path

# Allow importing from src (root directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from src.graphmemory.scene_graph_processor import SceneGraphProcessor
from src.utils.profiler import profiler

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def batch_process(config_path="config/default.yaml"):
    # 1. Load Config
    print(f"Loading config from {config_path}...")
    try:
        config = load_config(os.path.abspath(os.path.join(os.path.dirname(__file__), "../", config_path)))
    except FileNotFoundError:
        try:
             config = load_config(config_path)
        except Exception:
             config = {}

    if 'interaction' not in config:
        config['interaction'] = {}
    
    # 2. Setup Paths
    video_dir = config.get('data', {}).get('videos_dir', 'data/videos')
    video_path_pattern = os.path.join(os.path.dirname(__file__), "..", video_dir, "*.mp4")
    video_files = glob.glob(video_path_pattern)
    
    if not video_files:
        print(f"No videos found in {video_path_pattern}. Please add .mp4 files.")
        return

    # Device
    device = config.get('experiment', {}).get('device', 'auto')
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if torch.cuda.is_available(): device = 'cuda'
    print(f"Using device: {device}")
    
    # Initialize YOLO
    try:
        t0 = time.time()
        models_dir = config.get('data', {}).get('models_dir', 'data/models')
        model_name = config.get('detection', {}).get('model', 'yolo11n.pt')
        local_model_path = os.path.join(os.path.dirname(__file__), "..", models_dir, model_name)
        model_path = local_model_path if os.path.exists(local_model_path) else model_name
        model = YOLO(model_path)
        profiler.log("init_yolo", time.time() - t0)
    except Exception as e:
        print(f"Error loading YOLO: {e}")
        return

    logs_dir = config.get('data', {}).get('logs_dir', 'outputs/logs')
    logs_full_path = os.path.join(os.path.dirname(__file__), "..", logs_dir)
    os.makedirs(logs_full_path, exist_ok=True)

    # 3. Iterate Videos
    total_videos = len(video_files)
    for idx, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_filename)[0]
        log_path = os.path.join(logs_full_path, f"{video_name_no_ext}_events.json")
        timeline_path = os.path.join(logs_full_path, f"{video_name_no_ext}_timeline.png")
        
        print(f"Processing video {idx + 1}/{total_videos}: {video_filename}...")
        
        processor = SceneGraphProcessor(config)
        
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        
        conf_thresh = config.get('detection', {}).get('conf_threshold', 0.25)
        iou_thresh = config.get('detection', {}).get('iou_threshold', 0.45)
        persist = config.get('detection', {}).get('tracker_persist', True)
        
        t_detect_track = 0.0
        t_graph_update = 0.0
        
        while True:
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret: break
                
            # Detection & Tracking
            t1 = time.time()
            results = model.track(frame, persist=persist, device=device, verbose=False, conf=conf_thresh, iou=iou_thresh)
            t_detect_track += (time.time() - t1)
            
            detections = []
            for r in results:
                if r.boxes.id is not None:
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    for i, box in enumerate(boxes):
                        detections.append({
                            "id": int(ids[i]), "label": model.names[classes[i]], "bbox": tuple(box)
                        })
            
            # Graph Update
            t2 = time.time()
            timestamp = frame_idx / fps if fps > 0 else 0
            processor.update(detections, timestamp, frame_idx)
            t_graph_update += (time.time() - t2)
            
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  Frame {frame_idx}...")

        cap.release()
        
        # Save
        processor.save_events_to_json(log_path)
        processor.generate_timeline_plot(timeline_path)
        
        # Log profile
        profiler.log("detect_track", t_detect_track, video=video_filename)
        profiler.log("graph_update", t_graph_update, video=video_filename)
        print(f"  Completed. Det/Track: {t_detect_track:.2f}s, Graph: {t_graph_update:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    args = parser.parse_args()
    batch_process(args.config)

