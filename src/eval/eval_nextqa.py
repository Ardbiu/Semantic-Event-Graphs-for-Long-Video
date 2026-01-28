
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

from src.datasets.nextqa import NExTQA
from src.graphmemory.scene_graph_processor import SceneGraphProcessor
from src.temporal_graph import TemporalSceneGraph
from src.utils.profiler import Profiler
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class NExTQAEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Semantic Model for Candidate Scoring
        model_name = config.get('semantic_search', {}).get('model', 'all-MiniLM-L6-v2')
        print(f"Loading semantic model: {model_name}...")
        self.semantic_model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize Scene Graph Processor (for processing video frames)
        self.processor = SceneGraphProcessor(config)
        
        # Initialize Detector
        det_config = config.get('detection', {})
        self.detector_model = det_config.get('model_path', 'yolov8l.pt')
        print(f"Loading detector: {self.detector_model}...")
        from ultralytics import YOLO
        self.detector = YOLO(self.detector_model)
        
        # Cache for processed graphs (video_id -> TemporalSceneGraph)
        # In a full run, we might want to save these to disk.
        self.graph_cache = {}

    def get_or_build_graph(self, video_path, video_id):
        """
        Processes video to build a scene graph or loads it if already cached.
        """
        if video_id in self.graph_cache:
            return self.graph_cache[video_id]
        
        if not os.path.exists(video_path):
            # Special case for 'sample.mp4' smoke test if remapped
            return None

        print(f"Processing video: {video_id}")
        
        # Run Processor
        # Note: Processor logic usually iterates frames. 
        # For simplicity here, we assume batch_process style logic or call it directly.
        # But SceneGraphProcessor needs an iterator. 
        # For now, let's assume we can rely on pre-computed logs OR run passing the video path
        # Since we don't have a direct "process_video_file" method in Processor exposed purely,
        # we can simulate it or rely on batch_process.py having saved it?
        # Let's import the processing logic from process_video.py or similar if available.
        # Checking `scripts/process_video.py`... 
        # Actually, let's instantiate a fresh TSG from events.
        
        # Initialize Detector (Done in __init__)
        # det_config = self.config.get('detection', {})
        
        # Open Video
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Reset processor state
        self.processor = SceneGraphProcessor(self.config) 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_idx % self.config.get('detection', {}).get('sample_rate', 5) == 0:
                # Run Tracking
                results = self.detector.track(frame, persist=True, verbose=False)
                
                detections = []
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        if box.id is not None:
                            # Format: id, label, bbox
                            cls_id = int(box.cls[0].item())
                            label = results[0].names[cls_id]
                            bbox = box.xyxy[0].tolist()
                            if len(bbox) != 4:
                                print(f"DEBUG: Invalid bbox format: {bbox}")
                            
                            track_id = int(box.id[0].item())
                            
                            detections.append({
                                "id": track_id,
                                "label": label,
                                "bbox": bbox
                            })
                            
                timestamp = frame_idx / fps if fps > 0 else 0.0
                self.processor.update(detections, timestamp, frame_idx)
                
            frame_idx += 1
        cap.release()
        
        events = self.processor.get_events()
        
        # Build Temporal Graph
        tsg = TemporalSceneGraph()
        # Hack to load raw list instead of file
        tsg.all_events = events 
        # Manually trigger load similar to load_from_json
        for event in events:
            subj = event.get('subject')
            obj = event.get('object')
            if subj: tsg.graph.add_node(subj)
            if obj: tsg.graph.add_node(obj)
            if subj and obj:
                tsg.graph.add_edge(subj, obj, **event)
        
        self.graph_cache[video_id] = tsg
        return tsg

    def score_candidates(self, context_text, candidates):
        """
        Scores answer candidates against the context using cosine similarity.
        """
        # Embed context
        context_emb = self.semantic_model.encode(context_text, convert_to_tensor=True)
        
        # Embed candidates
        cand_embs = self.semantic_model.encode(candidates, convert_to_tensor=True)
        
        # Compute cosine similarities
        cur_scores = util.cos_sim(context_emb, cand_embs)[0]
        return cur_scores.cpu().numpy()

    def events_to_text(self, events):
        """Converts a list of events to a text description."""
        lines = []
        for ev in events:
            # e.g. "person-1 <interaction> object-2"
            lines.append(f"{ev['subject']} {ev['type']} {ev['object']}")
        return ". ".join(lines)

    def evaluate(self, dataset, max_samples=None):
        correct = 0
        total = 0
        
        # Group samples by video to minimize reprocessing
        samples_by_video = defaultdict(list)
        for i in range(len(dataset)):
            if max_samples and i >= max_samples: break
            sample = dataset[i]
            samples_by_video[sample['video_id']].append(sample)
            
        print(f"Evaluating on {len(samples_by_video)} videos...")
        
        for video_id, samples in tqdm(samples_by_video.items()):
            # 1. Build Graph
            # Use the path from the first sample
            video_path = samples[0]['video_path']
            
            # Check if video exists
            if not os.path.exists(video_path):
                # print(f"Skipping missing video: {video_id}")
                continue

            tsg = self.get_or_build_graph(video_path, video_id)
            if not tsg: continue
            
            for sample in samples:
                question = sample['question']
                candidates = sample['candidates']
                ground_truth = sample['answer_idx']
                
                # 2. Retrieve Context
                # Using 1-hop pruning strategy
                pruned_result = tsg.prune_and_retrieve(question, config=self.config)
                context_text = self.events_to_text(pruned_result.events)
                
                if not context_text: 
                    context_text = "No relevant events found."
                
                # 3. Score Candidates
                scores = self.score_candidates(context_text, candidates)
                prediction = np.argmax(scores)
                
                if prediction == ground_truth:
                    correct += 1
                total += 1
                
        accuracy = correct / total if total > 0 else 0.0
        print(f"\nFinal Results:")
        print(f"Total Samples evaluated: {total}")
        print(f"Accuracy: {accuracy:.2%}")
        
        # Save metrics
        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        }
        with open("outputs/mc_results.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    dataset = NExTQA(split=args.split)
    evaluator = NExTQAEvaluator(config)
    
    evaluator.evaluate(dataset, max_samples=args.limit)

if __name__ == "__main__":
    main()
