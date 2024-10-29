#!/bin/bash

#SBATCH --output=o_fold1_%j_cap.job
#SBATCH --error=e_fold1_%j_cap.job
#SBATCH --job-name=fold1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal

# Load the conda environment
source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

# Run the Python script
python combined_uncertain_fold1.py
