#!/bin/bash

#SBATCH --output=slurm_pyscript_%j_fold0.job
#SBATCH --error=slurm_error_%j_fold0.job
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --nice=10000

# Load the conda environment
source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

# Run the Python script
python combined_uncertain_fold1.py
