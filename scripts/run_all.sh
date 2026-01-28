#!/bin/bash
# Master script to run the full CVC 2026 experimental pipeline

set -e

# Configuration
CONFIG="config/default.yaml"
LOGS_DIR="outputs/logs"
RESULTS_FILE="outputs/final_results.json"
COST_FILE="outputs/cost_profile.csv"

echo "=========================================="
echo "    CVC 2026 Pipeline Smoke Test"
echo "=========================================="

# 1. Pipeline Execution (Detection -> Interaction -> Graph)
echo "[1/5] Running Video Pipeline (Detection & Graph Building)..."
python scripts/batch_process.py --config $CONFIG

# 2. Baselines
echo "[2/5] Running Baselines..."
python scripts/run_baselines.py --config $CONFIG

# 3. Method Evaluation (Ours)
echo "[3/5] Running Our Method Evaluation (SEG)..."
python scripts/run_full_evaluation.py --config $CONFIG

# 4. Parameter Sweeps (Optional - can be slow, maybe use --fast flag?)
echo "[4/5] Running Parameter Sweeps..."
# Only run if explicitly requested or ensuring coverage. 
# For smoke test, maybe skip or run minimal?
# We'll run it but user can interrupt.
# ./scripts/run_sweeps.sh 
echo "Skipping full sweeps for smoke test speed. Run ./scripts/run_sweeps.sh manually for Pareto data."

# 5. Generate Plots & Report
echo "[5/5] Generating Artifacts..."

# Generate Pareto Plot (needs sweep results, but we can plot what we have)
if [ -f "results/sweep_results.csv" ]; then
    python scripts/plot_pareto.py
fi

# Sanity Check for Artifacts
echo "------------------------------------------"
echo "Artifact Verification:"

if [ -f "$COST_FILE" ]; then
    echo " [OK] Cost Profile found: $COST_FILE"
else
    echo " [FAIL] Cost Profile MISSING"
fi

if [ -f "$RESULTS_FILE" ]; then
    echo " [OK] Final Results found: $RESULTS_FILE"
else
    echo " [FAIL] Final Results MISSING"
fi

# Function to count files
count_events=$(ls $LOGS_DIR/*_events.json 2>/dev/null | wc -l)
echo " [Info] Event logs generated: $count_events"

echo "=========================================="
echo "Smoke Test Complete."

# 5. Generate Plots & Tables
echo "Step 5: Generating artifacts..."
# python scripts/plot_results.py

echo "Done. Results in outputs/"
