#!/bin/bash
echo "Running profiling on NExT-QA validation set (limit 1 for test)..."
# Uses the instrumented eval_nextqa.py to generate cost_profile.csv
# Ensure videos are available in data/nextqa/videos/
python3 src/eval/eval_nextqa.py --limit 1 --split val
echo "Profile saved to outputs/cost_profile.csv"
