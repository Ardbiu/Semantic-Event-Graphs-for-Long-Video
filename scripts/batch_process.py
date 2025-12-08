import glob
import os
import torch
import sys
# Allow importing from src (root directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from src.graphmemory.scene_graph_processor import SceneGraphProcessor

def batch_process():
    # 1. Setup
    video_path_pattern = os.path.join(os.path.dirname(__file__), "../data/videos/*.mp4")
    video_files = glob.glob(video_path_pattern)
    if not video_files:
        print(f"No videos found in {video_path_pattern}. Please add .mp4 files.")
        return

    # Device selection (Mac/MPS support)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize YOLO
    model_path = os.path.join(os.path.dirname(__file__), "../data/models/yolo11n.pt")
    # If using local model path, ensure it exists or use simple name if relying on ultralytics cache. 
    # But we moved it to data/models, so let's point there.
    try:
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
             # Fallback to download or cache if file missing
            model = YOLO('yolo11n.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Ensure logs dir exists
    logs_dir = os.path.join(os.path.dirname(__file__), "../outputs/logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 2. Iterate Videos
    total_videos = len(video_files)
    for idx, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_filename)[0]
        log_path = os.path.join(logs_dir, f"{video_name_no_ext}_events.json")
        
        print(f"Processing video {idx + 1}/{total_videos}: {video_filename}...")
        
        # Initialize Processor
        # Focus on 'person' interacting with things, distance threshold 200px seems reasonable for standard resolutions
        processor = SceneGraphProcessor(focus_class="person", distance_threshold=200.0)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run Tracking
            # persist=True is crucial for ID tracking
            results = model.track(frame, persist=True, device=device, verbose=False)
            
            detections = []
            for r in results:
                if r.boxes.id is not None:
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, box in enumerate(boxes):
                        class_name = model.names[classes[i]]
                        detections.append({
                            "id": int(ids[i]),
                            "label": class_name,
                            "bbox": tuple(box)
                        })
            
            # Update Graph
            timestamp = frame_idx / fps if fps > 0 else 0
            processor.update(detections, timestamp, frame_idx)
            
            frame_idx += 1
            
            if frame_idx % 500 == 0:
                print(f"  Frame {frame_idx}...")

        cap.release()
        
        # Save Log
        processor.save_events_to_json(log_path)
        print(f"  Saved log to {log_path}")

if __name__ == "__main__":
    batch_process()
