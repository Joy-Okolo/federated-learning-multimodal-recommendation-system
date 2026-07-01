#!/bin/bash
#SBATCH --job-name=load_ecommerce_data
#SBATCH --output=logs/load_data_%j.out
#SBATCH --error=logs/load_data_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=compute

echo "Job started on $(hostname) at $(date)"
echo "Running on node with $(nproc) CPUs available"
echo "Testing network connectivity to huggingface.co..."
curl -s -o /dev/null -w "HTTP status: %{http_code}, time: %{time_total}s\n" https://huggingface.co || echo "curl failed -- possible network issue"

cd ~/fed_multimodal_rec/src/adapters

python -u amazon_reviews_adapter.py \
    --min_interactions 5 \
    --max_users 5000 \
    --out_interactions ../../data/interactions.parquet \
    --out_items ../../data/items.parquet

echo "Job finished at $(date)"
