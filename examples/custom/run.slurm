#!/bin/bash
#SBATCH --job-name=vllm_serve
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128g
#SBATCH --cpus-per-gpu=6
#SBATCH --partition=batch
#SBATCH --time=1:00:00
#SBATCH --output=vllm_serve_%j.out
#SBATCH --error=vllm_serve_%j.err

# Load any necessary modules here
# module load cuda/11.8

# Change to the script directory
cd /ibex/ai/home/alyafez/curator/examples/custom/

# Run the script
bash script.sh 2
