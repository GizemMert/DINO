#!/bin/bash

#SBATCH --output=slurm_pyscript_%j_cap.job
#SBATCH --error=slurm_error_%j_cap.job
#SBATCH --partition=cpu_p   Changed to CPU partition
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --nice=10000

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env
python single_cell_classification.py
