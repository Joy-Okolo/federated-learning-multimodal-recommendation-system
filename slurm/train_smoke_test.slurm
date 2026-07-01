#!/bin/bash
#SBATCH --job-name=train_smoke_test
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

mkdir -p logs

echo "Job started on $(hostname) at $(date)"
echo "Running on node with $(nproc) CPUs available"

cd ~/fed_multimodal_rec/src/training

python -u train.py \
    --epochs 1 \
    --batch_size 16

echo "Job finished at $(date)"
