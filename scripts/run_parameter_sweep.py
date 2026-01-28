import os
import sys
import yaml
import pandas as pd
import json
import itertools
from copy import deepcopy
import argparse

# Add scripts dir to path to import running functions
sys.path.append(os.path.dirname(__file__))
from batch_process import batch_process
from run_full_evaluation import main as run_evaluation

def save_temp_config(config, filename="config/temp_sweep.yaml"):
    with open(filename, 'w') as f:
        yaml.dump(config, f)
    return filename

def parse_results(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Calculate HyperGraph stats
    total = len(data)
    if total == 0:
        return 0, 0, 0
        
    correct = 0
    tokens = 0
    time = 0
    
    for row in data:
        res = row['models']['HyperGraph']
        if res['correct']:
            correct += 1
        tokens += res['tokens']
        time += res['time']
        
    return (correct / total) * 100, tokens / total, time / total

def run_sweeps(base_config_path="config/default.yaml"):
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # SWEEP PARAMETERS
    taus = [0.4, 0.6, 0.8]
    hops = [1, 2]
    top_ks = [3, 5, 10]
    
    # Override for quick test? 
    # taus = [0.6]
    # hops = [1, 2]
    
    results_list = []
    
    # Output file
    os.makedirs("results", exist_ok=True)
    csv_path = "results/sweep_results.csv"
    
    print("Starting Sweeps...")
    print(f"Taus: {taus}")
    print(f"Hops: {hops}")
    print(f"Top-Ks: {top_ks}")

    for tau in taus:
        print(f"\n--- Sweeping Threshold tau={tau} ---")
        
        # 1. Update Interaction Config
        current_config = deepcopy(base_config)
        current_config['interaction']['thresholds']['tau_start'] = tau
        current_config['interaction']['thresholds']['tau_end'] = max(0.1, tau - 0.2) # Auto-adjust hysteresis
        
        # Save temp config for batch inputs
        temp_config_path = save_temp_config(current_config)
        
        # 2. Run Batch Process (Generate Logs)
        print("Regenerating Event Logs...")
        # We need to make sure batch_process writes to a temp logs dir or overwrites?
        # Overwriting is fine for sweeps if we don't need to keep them all simultaneously.
        # But to be safe, let's use the default logs dir.
        try:
            batch_process(temp_config_path)
        except Exception as e:
            print(f"Error in batch_process: {e}")
            continue

        for hop, k in itertools.product(hops, top_ks):
            print(f"  Testing Graph: hop={hop}, top_k={k}")
            
            # 3. Update Graph Config
            current_config['graph']['pruning']['hop_depth'] = hop
            current_config['graph']['pruning']['top_k_neighbors'] = k
            
            # Save again
            temp_config_path = save_temp_config(current_config)
            
            # 4. Run Evaluation
            try:
                run_evaluation(temp_config_path)
                
                # 5. Parse Results
                acc, toks, duration = parse_results(current_config['data']['results_file'])
                
                print(f"    Result: Acc={acc:.1f}%, Toks={toks:.0f}, Time={duration:.2f}s")
                
                results_list.append({
                    "tau": tau,
                    "hop_depth": hop,
                    "top_k": k,
                    "accuracy": acc,
                    "avg_tokens": toks,
                    "avg_time": duration
                })
                
                # Save intermediate
                df = pd.DataFrame(results_list)
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error in evaluation: {e}")

    print(f"\nSweeps Complete. Results saved to {csv_path}")

if __name__ == "__main__":
    run_sweeps()
