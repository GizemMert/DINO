#!/bin/bash

#SBATCH --output=o_fold0_%j_cap.job
#SBATCH --error=e_fold0_%j_cap.job
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --nice=10000

# Load the conda environment
source /home/aih/gizem.mert/tools/apps/mamba/etc/profile.d/conda.sh
conda activate myenv

# Run the Python script
python combined_uncertain_fold0.py
