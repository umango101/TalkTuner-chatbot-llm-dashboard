#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=openai_priorities
#SBATCH --output=logs/%j.log

module load python 
source activate talktuner-gpu
echo "Conda env: $CONDA_DEFAULT_ENV"
python -u notebooks/dataset_generation/openai_priorities_dataset_generation.py
