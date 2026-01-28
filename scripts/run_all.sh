#!/bin/bash
# Master script to run the full CVC 2026 experimental pipeline

set -e

echo "Starting CVC 2026 Experiment Suite..."

# 1. Process Videos -> Generate Event Logs (with new Interaction logic)
echo "Step 1: Processing videos to generate semantic event logs..."
# python scripts/batch_process.py --config config/default.yaml

# 2. Run Baselines
echo "Step 2: Running Baselines..."
# python scripts/run_baselines.py --config config/default.yaml

# 3. Run Our Method (SEG/TSG)
echo "Step 3: Running Semantic Event Graph method..."
# python scripts/run_seg_method.py --config config/default.yaml

# 4. Evaluation
echo "Step 4: Running Evaluation with Judge Model..."
# python scripts/evaluate_all.py --config config/default.yaml

# 5. Generate Plots & Tables
echo "Step 5: Generating artifacts..."
# python scripts/plot_results.py

echo "Done. Results in outputs/"
