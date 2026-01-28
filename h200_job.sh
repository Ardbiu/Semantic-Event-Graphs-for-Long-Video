#!/bin/bash
#SBATCH --job-name=seg_eval
#SBATCH --output=logs/seg_eval_%j.out
#SBATCH --error=logs/seg_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Initialize configuration
source /etc/profile
module load anaconda3/2024.02-1
conda activate seg_env

echo "Job started on $(hostname) at $(date)"
echo "GPU Info:"
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs outputs/logs

# Run the full pipeline smoke test
./scripts/run_all.sh

echo "Job finished at $(date)"
