import json
import os
import sys
import glob

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.temporal_graph import TemporalSceneGraph

def events_to_narrative(events):
    narrative = []
    for ev in events:
        line = f"[{ev.get('timestamp', 0):.1f}s] {ev.get('subject')} {ev.get('type')} {ev.get('object')}"
        narrative.append(line)
    return "\n".join(narrative)

def check_logs():
    print("=" * 60)
    print("1. LOG FILE ANALYSIS")
    print("=" * 60)
    
    log_dir = os.path.join(os.path.dirname(__file__), "../outputs/logs")
    log_files = glob.glob(os.path.join(log_dir, "*_events.json"))
    for log_path in sorted(log_files):
        with open(log_path, 'r') as f:
            events = json.load(f)
        filename = os.path.basename(log_path)
        print(f"  {filename}: {len(events)} events")
    
    # Print sample events from a non-empty file
    for log_path in log_files:
        with open(log_path, 'r') as f:
            events = json.load(f)
        if len(events) > 0:
            print(f"\n  Sample events from {os.path.basename(log_path)}:")
            for ev in events[:3]:
                print(f"    {ev}")
            break

def debug_evaluation_logic():
    print("\n" + "=" * 60)
    print("2. EVALUATION LOGIC DEBUG")
    print("=" * 60)
    
    # Load benchmark
    benchmark_path = os.path.join(os.path.dirname(__file__), "../outputs/benchmark_data.json")
    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)
    
    # Find first valid question
    test_item = None
    test_question = None
    for item in benchmark_data:
        for qa in item["qa_pairs"]:
            if "empty" not in qa["q"].lower():
                test_item = item
                test_question = qa
                break
        if test_question:
            break
    
    if not test_question:
        print("  ERROR: No valid questions found!")
        return

    source_log = test_item["source_log"]
    log_path = os.path.join(os.path.dirname(__file__), "../outputs/logs", source_log)
    
    print(f"  Source Log: {source_log}")
    print(f"  Question: {test_question['q'][:80]}...")
    
    # Load Events
    with open(log_path, 'r') as f:
        all_events = json.load(f)
    
    print(f"\n  Total Events in Log: {len(all_events)}")
    
    # Simulate Short Context
    if all_events:
        max_time = max(ev.get("timestamp", 0) for ev in all_events)
        cutoff = max_time - 30.0
        recent_events = [ev for ev in all_events if ev.get("timestamp", 0) >= cutoff]
    else:
        recent_events = []
    
    short_narrative = events_to_narrative(recent_events)
    long_narrative = events_to_narrative(all_events)
    
    print(f"\n  SHORT CONTEXT:")
    print(f"    Events Used: {len(recent_events)}")
    print(f"    Narrative Char Length: {len(short_narrative)}")
    
    print(f"\n  LONG CONTEXT:")
    print(f"    Events Used: {len(all_events)}")
    print(f"    Narrative Char Length: {len(long_narrative)}")
    
    # Check if they are the same
    if len(short_narrative) == len(long_narrative):
        print("\n  ⚠️  WARNING: Short and Long narratives are IDENTICAL LENGTH!")
        print("      This suggests the log file has very few events, or all events")
        print("      fall within the last 30 seconds of the video.")
    else:
        print(f"\n  ✔ Narratives have DIFFERENT lengths (as expected).")
        print(f"    Difference: {len(long_narrative) - len(short_narrative)} chars")

def check_pruning():
    print("\n" + "=" * 60)
    print("3. HYPERGRAPH PRUNING DEBUG")
    print("=" * 60)
    
    # Load benchmark
    benchmark_path = os.path.join(os.path.dirname(__file__), "../outputs/benchmark_data.json")
    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)
    
    # Find first valid question
    test_item = None
    test_question = None
    for item in benchmark_data:
        for qa in item["qa_pairs"]:
            if "empty" not in qa["q"].lower():
                test_item = item
                test_question = qa
                break
        if test_question:
            break
    
    source_log = test_item["source_log"]
    log_path = os.path.join(os.path.dirname(__file__), "../outputs/logs", source_log)
    
    tsg = TemporalSceneGraph()
    tsg.load_from_json(log_path)
    
    question = test_question['q']
    pruned_result = tsg.prune_and_retrieve(question)
    
    print(f"  Question: {question[:80]}...")
    print(f"  Pruned Events Retrieved: {len(pruned_result.events)}")
    print(f"  Compression Ratio: {pruned_result.compression_ratio:.1%}")
    
    if pruned_result.events:
        print(f"  Sample Pruned Event: {pruned_result.events[0]}")

if __name__ == "__main__":
    print("\n🔍 PIPELINE DIAGNOSTIC REPORT\n")
    check_logs()
    debug_evaluation_logic()
    check_pruning()
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
