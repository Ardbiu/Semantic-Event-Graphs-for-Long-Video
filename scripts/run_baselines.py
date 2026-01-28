import os
import sys
import json
import yaml
import time
import argparse

# Allow importing from src (root directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.baselines.video_llm import VideoLLMBaseline
from src.baselines.retrieval import RetrievalBaseline
from src.baselines.summarization import SummarizationBaseline
from src.llm_judge import evaluate_accuracy

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_baselines(config_path="config/default.yaml"):
    print(f"Loading config from {config_path}...")
    try:
        config = load_config(config_path)
    except:
        config = {}

    benchmark_file = config.get('data', {}).get('benchmark_file', 'outputs/benchmark_data.json')
    if not os.path.exists(benchmark_file):
        # Try relative
        benchmark_file = os.path.join(os.path.dirname(__file__), "../", benchmark_file)

    if not os.path.exists(benchmark_file):
        print(f"Benchmark file not found: {benchmark_file}")
        return

    with open(benchmark_file, "r") as f:
        benchmark_data = json.load(f)

    # Initialize Baselines
    baselines = {}
    
    # 1. VideoLLM
    try:
        baselines['VideoLLM'] = VideoLLMBaseline()
    except Exception as e:
        print(f"Skipping VideoLLM baseline: {e}")

    # 2. Retrieval
    try:
        baselines['Retrieval'] = RetrievalBaseline()
    except Exception as e:
        print(f"Skipping Retrieval baseline: {e}")

    # 3. Summarization
    try:
        baselines['Summarization'] = SummarizationBaseline()
    except Exception as e:
        print(f"Skipping Summarization baseline: {e}")

    if not baselines:
        print("No baselines successfully initialized.")
        return

    results = []
    
    # Iterate Videos
    for item in benchmark_data:
        source_log = item["source_log"] # e.g. "video1_events.json"
        video_name = source_log.replace("_events.json", ".mp4")
        
        # Resolving Video Path
        video_dir = config.get('data', {}).get('videos_dir', 'data/videos')
        video_path = os.path.join(os.path.dirname(__file__), "..", video_dir, video_name)
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
            
        qa_pairs = item["qa_pairs"]
        print(f"\nProcessing {video_name} ({len(qa_pairs)} questions)...")

        r1_captions = None
        if 'Retrieval' in baselines:
            print("  Generating captions for Retrieval Baseline...")
            try:
                r1_captions = baselines['Retrieval'].caption_frames(video_path)
            except Exception as e:
                print(f"  Retrieval captioning failed: {e}")

        for qa in qa_pairs:
            question = qa["q"]
            ground_truth = qa["a"]
            
            row = {
                "question": question,
                "ground_truth": ground_truth,
                "models": {}
            }
            
            print(f"  Q: {question[:40]}...")

            # Run Each Baseline
            for name, model in baselines.items():
                start_time = time.time()
                try:
                    ans = "Error"
                    if name == 'Retrieval' and r1_captions:
                        # R1 specific flow
                        retrieved = model.retrieve(r1_captions, question)
                        # Construct context
                        context = "\n".join([f"[{c['timestamp']:.1f}s] {c['text']}" for c in retrieved])
                        
                        import google.generativeai as genai
                        model_gen = genai.GenerativeModel('gemini-1.5-flash')
                        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                        res = model_gen.generate_content(prompt)
                        ans = res.text.strip()
                        
                    else:
                        ans = model.answer_question(video_path, question)
                        
                    duration = time.time() - start_time
                    
                    # Evaluate
                    is_correct = evaluate_accuracy(question, ground_truth, ans)
                    
                    row["models"][name] = {
                        "answer": ans,
                        "time": duration,
                        "correct": is_correct
                    }
                    print(f"    {name}: {'✔' if is_correct else '✘'} ({duration:.2f}s)")
                    
                except Exception as e:
                    print(f"    {name}: Failed ({e})")
                    row["models"][name] = {"error": str(e)}

            results.append(row)

    # Save
    out_file = config.get('data', {}).get('baseline_results_file', 'outputs/baseline_results.json')
    out_path = os.path.join(os.path.dirname(__file__), "..", out_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config")
    args = parser.parse_args()
    
    run_baselines(args.config)
