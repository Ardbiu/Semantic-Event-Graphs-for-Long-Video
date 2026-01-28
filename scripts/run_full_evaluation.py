import json
import os
import time
import sys
import yaml
import argparse

# Allow importing from src (root directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai
from src.temporal_graph import TemporalSceneGraph
from src.llm_judge import evaluate_accuracy

# Setup API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY environment variable not found.")
else:
    genai.configure(api_key=api_key)

MODEL_NAME = 'gemini-2.5-flash'

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(config=None):
    model_name = MODEL_NAME
    if config:
        model_name = config.get('evaluation', {}).get('answer_model', MODEL_NAME)
    
    try:
        return genai.GenerativeModel(model_name)
    except:
        return genai.GenerativeModel('gemini-1.5-flash')

def events_to_narrative(events):
    narrative = []
    for ev in events:
        # Include confidence if available
        conf_str = ""
        if 'confidence' in ev:
             conf_str = f" (conf:{ev['confidence']:.2f})"
        
        line = f"[{ev.get('timestamp', 0):.1f}s] {ev.get('subject')} {ev.get('type')} {ev.get('object')}{conf_str}"
        narrative.append(line)
    return "\n".join(narrative)

def generate_answer(narrative, question, config=None):
    model = get_model(config)
    prompt = f"""
    You are an expert video analyst.
    Task: Answer the question based strictly on the provided Event Log.
    
    Instructions:
    1. Scan the log for events related to the entities in the question.
    2. Note the timestamps and sequence of actions.
    3. Think step-by-step to derive the answer.
    
    Event Log:
    {narrative}
    
    Question: {question}
    """
    
    extra_sleep = 0
    for attempt in range(3):
        try:
            # Count tokens
            token_count = model.count_tokens(prompt).total_tokens
            
            start_time = time.time()
            response = model.generate_content(prompt)
            text = response.text.strip()
            duration = time.time() - start_time
            
            return text, token_count, duration
        except Exception as e:
            print(f"    ⚠️  Error generating answer (Attempt {attempt+1}/3): {e}")
            if "Deadline" in str(e) or "504" in str(e) or "429" in str(e):
                time.sleep(5 + extra_sleep)
                extra_sleep += 5
                continue
            return "Error", 0, 0
    return "Error (Timeout)", 0, 0

def judge_with_retry(question, ground_truth, ans):
    # Retry logic for the judge
    for attempt in range(3):
        try:
            return evaluate_accuracy(question, ground_truth, ans)
        except Exception:
            time.sleep(2)
    return False

# Strategy 1: Short Context (Last 30s)
def run_short_context(events, question, config=None):
    if not events:
        return "No events", 0, 0
    max_time = max((ev.get("timestamp", 0) for ev in events), default=0)
    cutoff = max_time - 30.0
    recent_events = [ev for ev in events if ev.get("timestamp", 0) >= cutoff]
    narrative = events_to_narrative(recent_events)
    return generate_answer(narrative, question, config)

# Strategy 2: Long Context (All Events)
def run_long_context(events, question, config=None):
    narrative = events_to_narrative(events)
    return generate_answer(narrative, question, config)

# Strategy 3: HyperGraph (Pruned)
def run_hypergraph(graph, question, config=None):
    # Pass config to prune_and_retrieve so it can use hop_depth/top_k
    pruned_result = graph.prune_and_retrieve(question, config=config)
    
    # DEBUG PRINT as requested
    print(f"    DEBUG: Pruned from {len(graph.all_events)} events to {len(pruned_result.events)} events.")
    
    narrative = events_to_narrative(pruned_result.events)
    return generate_answer(narrative, question, config)

import numpy as np
from collections import defaultdict

def compute_bootstrap_ci(scores, n_bootstrap=1000, ci=95):
    """
    Compute Bootstrap 95% CI for a list of binary scores (0/1).
    """
    if not scores:
        return 0, 0, 0
    scores = np.array(scores)
    means = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(resample.mean())
    
    means = np.array(means)
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return means.mean(), lower, upper

def main(config_path="config/default.yaml"):
    print(f"Loading config from {config_path}...")
    try:
        config = load_config(config_path)
    except Exception:
        config = {}
        print("Using default/empty config.")

    # Protocol Verification Box
    print("\n" + "#"*60)
    print(" PROTOCOL VERIFICATION")
    print("#"*60)
    print(f"Answer Model: {config.get('evaluation', {}).get('answer_model', 'gemini-1.5-flash')}")
    print(f"Judge Model:  {MODEL_NAME}") 
    print(f"Sampling:     Full Context (Method) vs 8-frame (VideoLLM) vs Caption (Retrieval)")
    print("#"*60 + "\n")

    # Override results file path from config
    results_file = config.get('data', {}).get('results_file', 'outputs/final_results.json')
    benchmark_file = config.get('data', {}).get('benchmark_file', 'outputs/benchmark_data.json')
    
    # Also support old path if config missing
    if not os.path.exists(benchmark_file):
        benchmark_file = os.path.join(os.path.dirname(__file__), "../outputs/benchmark_data.json")

    if not os.path.exists(benchmark_file):
        print(f"benchmark_data.json not found at {benchmark_file}")
        return

    with open(benchmark_file, "r") as f:
        benchmark_data = json.load(f)

    graph_cache = {}
    results = []
    total_q = 0
    
    # Store per-question correctness for CI
    models_scores = defaultdict(list)
    stats = defaultdict(lambda: {"correct": 0, "tokens": 0, "time": 0})

    logs_dir = config.get('data', {}).get('logs_dir', 'outputs/logs')
    # Resolve relative path
    logs_full_path = os.path.join(os.path.dirname(__file__), "..", logs_dir)

    for item in benchmark_data:
        source_log = item["source_log"]
        qa_pairs = item["qa_pairs"]
        log_path = os.path.join(logs_full_path, source_log)
        
        if not os.path.exists(log_path):
             # Try simple name if path construction failed
             log_path_simple = os.path.join(os.path.dirname(__file__), "../outputs/logs", source_log)
             if os.path.exists(log_path_simple):
                 log_path = log_path_simple
             else:
                 print(f"Log file not found: {log_path}, skipping...")
                 continue
            
        print(f"\nProcessing {source_log} ({len(qa_pairs)} questions)...")
        
        if source_log not in graph_cache:
            tsg = TemporalSceneGraph()
            tsg.load_from_json(log_path)
            graph_cache[source_log] = tsg
        
        tsg = graph_cache[source_log]
        all_events = tsg.all_events
        
        for qa in qa_pairs:
            question = qa["q"]
            ground_truth = qa["a"]
            
            if "empty" in question.lower():
                continue

            total_q += 1
            print(f"  Q{total_q}: {question[:50]}...")

            # Run 3 Models
            models = [
                ("Short", lambda: run_short_context(all_events, question, config)),
                ("Long", lambda: run_long_context(all_events, question, config)),
                ("HyperGraph", lambda: run_hypergraph(tsg, question, config))
            ]
            
            row = {
                "question": question,
                "ground_truth": ground_truth,
                "models": {}
            }
            
            for name, func in models:
                ans, tokens, duration = func()
                is_correct = judge_with_retry(question, ground_truth, ans)
                
                row["models"][name] = {
                    "answer": ans,
                    "tokens": tokens,
                    "time": duration,
                    "correct": is_correct
                }
                
                if is_correct:
                    stats[name]["correct"] += 1
                    models_scores[name].append(1)
                else:
                    models_scores[name].append(0)
                
                stats[name]["tokens"] += tokens
                stats[name]["time"] += duration
                
                print(f"    {name}: {'✔' if is_correct else '✘'} ({tokens} toks, {duration:.2f}s)")
            
            results.append(row)

    # Save Detailed Results
    output_path = os.path.join(os.path.dirname(__file__), "..", results_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'95% CI':<15} | {'Avg Tokens':<12} | {'Avg Time (s)':<12}")
    print("-" * 80)
    
    if total_q > 0:
        for name in ["Short", "Long", "HyperGraph"]:
            accuracy = (stats[name]["correct"] / total_q) * 100
            avg_tokens = stats[name]["tokens"] / total_q
            avg_time = stats[name]["time"] / total_q
            
            # Compute CI
            _, lower, upper = compute_bootstrap_ci(models_scores[name])
            ci_str = f"[{lower*100:.1f}, {upper*100:.1f}]"
            
            print(f"{name:<15} | {accuracy:6.1f}%    | {ci_str:<15} | {avg_tokens:10.1f}   | {avg_time:10.2f}")
    print("="*80)
    print(f"Detailed results saved to {results_file}. Total Questions: {total_q}")
    print(f"Judge Model: {MODEL_NAME}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)
