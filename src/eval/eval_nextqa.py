
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
from src.utils.profiler import profiler
import yaml
import time
import google.generativeai as genai


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

        # Configure Gemini if needed
        if config.get('gemini_api_key'):
            genai.configure(api_key=config.get('gemini_api_key'))
            model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
            print(f"Initializing Gemini Model: {model_name}")
            self.gemini_model = genai.GenerativeModel(model_name)
        else:
            self.gemini_model = None

    def ask_gemini(self, context_text, question, candidates):
        """
        Queries Gemini with the graph context to answer the question using Structured Reasoning.
        Returns: (predicted_index, token_count, reasoning)
        """
        if not self.gemini_model:
            raise ValueError("Gemini API key not configured")

        prompt = f"""
        > **System Role:** You are a Vision-Language Reasoning Engine specialized in temporal logic and long-form video understanding.
        > **Context:** You are provided with a **Semantic Event Graph** extracted from a video. This graph consists of discrete events (nodes) and their temporal/causal relationships (edges).
        
        > **Retrieved Subgraph Data:**
        {context_text}
        
        > **Task:** Based **strictly** on the retrieved graph evidence above, answer the following question.
        > 1. Analyze the sequence of events.
        > 2. Identify the specific timestamps or event IDs that provide the answer.
        > 3. Select the correct option from the choices provided.
        
        **Question:** {question}
        
        **Options:**
        0: {candidates[0]}
        1: {candidates[1]}
        2: {candidates[2]}
        3: {candidates[3]}
        4: {candidates[4]}
        
        **Response Format (JSON):**
        {{
            "reasoning": "A brief step-by-step explanation of the temporal logic used.",
            "evidence_nodes": ["List the IDs of the events used to find the answer"],
            "answer_idx": 1,
            "confidence": "Scale 1-10"
        }}
        
        Respond with VALID JSON ONLY. Do not include markdown formatting like ```json.
        """
        
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt)
                
                # Check for blocking (safety)
                if response.prompt_feedback.block_reason:
                    return -1, 0, f"Blocked: {response.prompt_feedback.block_reason}"
                    
                text = response.text.replace('```json', '').replace('```', '').strip()
                data = json.loads(text)
                
                predicted_idx = int(data.get('answer_idx', -1))
                reasoning = data.get('reasoning', '')
                
                input_tokens = self.gemini_model.count_tokens(prompt).total_tokens
                return predicted_idx, input_tokens, reasoning

            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    delay = base_delay * (2 ** attempt)
                    print(f"Gemini 429 Rate Limit. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"Gemini Error or JSON Parse Error: {e} | Text: {response.text if 'response' in locals() else 'N/A'}")
                    return -1, 0, f"Error: {e}"
        
        return -1, 0, "Max Retries Exceeded (429)"

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
                t0 = time.time()
                results = self.detector.track(frame, persist=True, verbose=False)
                profiler.log('detection', time.time() - t0, video=video_id)
                
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
                t0 = time.time()
                self.processor.update(detections, timestamp, frame_idx)
                profiler.log('graph_update', time.time() - t0, video=video_id)
                # print(f"DEBUG: Frame {frame_idx} Detections: {len(detections)}")
                
            frame_idx += 1
        cap.release()
        
        events = self.processor.event_log
        print(f"DEBUG: Total Generated Events: {len(events)}")
        if len(events) > 0:
            print(f"DEBUG: Sample Event: {events[0]}")
        
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
        """Converts a list of events to a Graph-Aware text description."""
        lines = []
        for i, ev in enumerate(events):
            # Format: EVENT_ID: [001] | TIME: 00:05 | ACTION: ...
            ts = ev.get('timestamp', 0.0)
            time_str = f"{int(ts//60):02d}:{int(ts%60):02d}"
            
            lines.append(f"EVENT_ID: [{i}] | TIME: {time_str} | ACTION: {ev['subject']} {ev['type']} {ev['object']}")
            
            # Add simple 'next' relation for flow
            if i < len(events) - 1:
                 lines.append(f"RELATION: [{i}] is followed by [{i+1}]")
                 
        return "\n".join(lines)

    def apply_tpa(self, context_events, all_events):
        """
        Temporal Path Anchoring (TPA): Uses the graph structure to bridge gaps.
        If two retrieved events are within a threshold (e.g. 20s), retrieve the chain of events between them.
        """
        if not context_events: return []
        
        # Sort by timestamp
        context_events = sorted(context_events, key=lambda x: x.get('timestamp', 0))
        
        # Build lookup for all events by timestamp
        # all_events is likely a list. Ideally we sort it too.
        full_timeline = sorted(all_events, key=lambda x: x.get('timestamp', 0))
        
        tpa_events = []
        GAP_THRESHOLD = 20.0 # seconds
        
        for i in range(len(context_events)):
            curr = context_events[i]
            tpa_events.append(curr)
            
            if i < len(context_events) - 1:
                next_ev = context_events[i+1]
                t1 = curr.get('timestamp', 0)
                t2 = next_ev.get('timestamp', 0)
                
                # If gap is small enough, fill it (Causal Bridge)
                if 0 < (t2 - t1) < GAP_THRESHOLD:
                    # Find bridges
                    bridges = [e for e in full_timeline if t1 < e.get('timestamp', 0) < t2]
                    # Add them if not too many (avoid exploding context)
                    if len(bridges) < 5:
                        for b in bridges:
                            b['_is_bridge'] = True # Marker
                        tpa_events.extend(bridges)
                        
        # Remove duplicates
        unique = []
        seen = set()
        for ev in tpa_events:
            # Create a simple hashable ID
            sig = f"{ev.get('timestamp')}_{ev.get('subject')}"
            if sig not in seen:
                seen.add(sig)
                unique.append(ev)
                
        return sorted(unique, key=lambda x: x.get('timestamp', 0))

    def evaluate(self, dataset, max_samples=None, use_gemini=False):
        correct = 0
        total = 0
        total_tokens = 0
        
        # Gather all samples first
        all_samples = [dataset[i] for i in range(len(dataset))]
        
        # Filter for "Stress Test" Subset (Causal/Temporal/Long)
        stress_samples = []
        for s in all_samples:
            q_lower = s['question'].lower()
            if q_lower.startswith('why') or q_lower.startswith('how'):
                stress_samples.append(s)
        
        print(f"Filtered {len(stress_samples)}/{len(all_samples)} samples for Causal/Temporal Stress Test")
        samples_to_use = stress_samples
        
        # Group samples by video to minimize reprocessing
        samples_by_video = defaultdict(list)
        sample_count = 0
        for sample in samples_to_use:
            if max_samples and sample_count >= max_samples: break
            samples_by_video[sample['video_id']].append(sample)
            sample_count += 1
            
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
                answer_str = candidates[ground_truth]
                
                # 2. Retrieve Context
                # Using 1-hop pruning strategy
                t0 = time.time()
                pruned_result = tsg.prune_and_retrieve(question, config=self.config)
                print(f"DEBUG: Retrieved {len(pruned_result.events)} events before TPA")
                
                # Apply TPA (Novelty!)
                context_events = pruned_result.events
                if hasattr(tsg, 'all_events') and tsg.all_events:
                    context_events = self.apply_tpa(pruned_result.events, tsg.all_events)
                
                print(f"DEBUG: Context Events after TPA: {len(context_events)}")
                
                context_text = self.events_to_text(context_events)
                
                # LEAKAGE CHECK
                if answer_str.lower() in context_text.lower():
                    print(f"WARNING: Potential Leakage in Video {video_id}. Answer '{answer_str}' found in context.")
                
                # 3. Score Candidates
                t0 = time.time()
                
                reasoning = ""
                prediction = -1
                input_tokens = 0
                
                if use_gemini:
                    # Graph RAG Mode
                    try:
                        prediction, input_tokens, reasoning = self.ask_gemini(context_text, question, candidates)
                        total_tokens += input_tokens
                        # Fallback if Gemini fails
                        if prediction == -1:
                            scores = self.score_candidates(context_text, candidates)
                            prediction = np.argmax(scores)
                            reasoning = "Fallback to Embedding Similarity (Gemini Failure)"
                    except Exception as e:
                        print(f"Gemini Exception in evaluate: {e}")
                else:
                    # Standard Embedding Mode
                    scores = self.score_candidates(context_text, candidates)
                    prediction = np.argmax(scores)
                    
                # Log Result (Success or Failure)
                log_entry = {
                    "video_id": video_id,
                    "question": question,
                    "ground_truth": answer_str,
                    "prediction": candidates[prediction] if 0 <= prediction < 5 else "Error",
                    "reasoning": reasoning,
                    "status": "CORRECT" if prediction == ground_truth else "WRONG",
                    "context_snippet": context_text[:300]
                }
                
                # Use line buffering or flush
                with open("outputs/detailed_results.jsonl", "a", buffering=1) as f:
                    f.write(json.dumps(log_entry) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                if prediction == ground_truth:
                    correct += 1
                total += 1
                
                print(f"Sample {total} | Video {video_id}: Pred {prediction} vs GT {ground_truth} -> {'CORRECT' if prediction == ground_truth else 'WRONG'}")
                print(f"Running Accuracy: {correct/total:.2%}")

                profiler.log('scoring', time.time() - t0, video=video_id)
                
        accuracy = correct / total if total > 0 else 0.0
        print(f"\nFinal Results:")
        print(f"Total Samples evaluated: {total}")
        print(f"Accuracy: {accuracy:.2%}")
        
        # Save metrics
        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "avg_tokens": total_tokens / total if total > 0 else 0
        }
        }
        with open("outputs/mc_results.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument('--use_gemini', action='store_true', help="Use Gemini for answering (Graph RAG)")
    args = parser.parse_args()
    
    # Inject API key into config if present in env
    args.config_data = load_config(args.config)
    if os.environ.get('GEMINI_API_KEY'):
        args.config_data['gemini_api_key'] = os.environ.get('GEMINI_API_KEY')
    
    dataset = NExTQA(split=args.split)
    evaluator = NExTQAEvaluator(args.config_data)
    
    evaluator.evaluate(dataset, max_samples=args.limit, use_gemini=args.use_gemini)

if __name__ == "__main__":
    main()
