#!/bin/bash
echo "Running profiling on sample video..."
# Run batch process on a subset or single video to profile
python scripts/batch_process.py --config config/default.yaml
echo "Profile saved to outputs/cost_profile.csv"
