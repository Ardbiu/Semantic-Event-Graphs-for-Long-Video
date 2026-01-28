import json
import os
import argparse
import sys
import glob

def inspect_events(logs_dir="outputs/logs", num_examples=10):
    print(f"Inspecting events in {logs_dir}...")
    log_files = glob.glob(os.path.join(logs_dir, "*_events.json"))
    
    if not log_files:
        print("No event logs found.")
        return

    print(f"{'Time':<10} | {'Subject':<15} | {'Object':<15} | {'Type':<10} | {'Score':<6} | {'Evidence (Prox/Mot/Sem)'}")
    print("-" * 100)
    
    count = 0
    for log_path in log_files:
        with open(log_path, 'r') as f:
            events = json.load(f)
            
        for ev in events:
            # Check for evidence field
            evidence = ev.get('evidence', {})
            conf = ev.get('confidence', 0.0)
            
            # extract individual components if available
            p_score = evidence.get('proximity', 0.0)
            m_score = evidence.get('motion', 0.0)
            s_score = evidence.get('semantic', 0.0)
            
            # Format Evidence String
            # "P:0.9 M:0.1 S:0.8"
            ev_str = f"P:{p_score:.2f} M:{m_score:.2f} S:{s_score:.2f}"
            
            print(f"{ev['timestamp']:<10.2f} | {ev['subject']:<15} | {ev['object']:<15} | {ev['type']:<10} | {conf:<6.2f} | {ev_str}")
            
            count += 1
            if count >= num_examples:
                break
        if count >= num_examples:
            break

    print("-" * 100)
    print("Verification:")
    if count == 0:
        print("[FAIL] No events found to inspect.")
    else:
        # Check if evidence was actually populated (not just 0.0 defaults)
        if s_score == 0.0 and m_score == 0.0:
             print("[WARN] Semantic/Motion scores seem to be 0.0. Verify InteractionScorer logic.")
        else:
             print("[OK] Evidence fields present and populated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="outputs/logs")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    
    inspect_events(args.logs_dir, args.n)
