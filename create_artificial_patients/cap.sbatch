#!/bin/bash

#SBATCH --output=slurm_pyscript_%j_cap.job
#SBATCH --error=slurm_error_%j_cap.job
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=72:00:00
#SBATCH --nice=10000

# Load the conda environment
source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

# Run the Python script
python 3_shuffle_images_distribution.py
