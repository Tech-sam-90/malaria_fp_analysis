#!/bin/bash
#SBATCH --account=rpp-uanazodo_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=model_b_train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Load modules
module load gcc opencv python/3.11 cuda/12.2

# Activate virtual environment
source ~/scratch/venv/bin/activate

# Environment variables
export WANDB_API_KEY=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
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

# Run training
cd ~/scratch/Practicum_Main_Project
python Notebooks/model_b_training.py

echo "Job finished at: $(date)"
