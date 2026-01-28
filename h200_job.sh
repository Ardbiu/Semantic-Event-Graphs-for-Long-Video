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
# Explicitly source conda.sh from the path found on the cluster
source /nfs/software001/home/software-r8-x86_64/spack-20230328/opt/spack/linux-rocky8-x86_64/gcc-8.5.0/anaconda3-2022.05-auh4o3tsby7ze6q6v3stn2hhvvnpoy5f/etc/profile.d/conda.sh
conda activate seg_env

echo "Job started on $(hostname) at $(date)"
echo "GPU Info:"
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs outputs/logs

# Run the full pipeline smoke test
./scripts/run_all.sh

echo "Job finished at $(date)"
