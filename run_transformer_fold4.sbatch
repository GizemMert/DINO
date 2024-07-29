#!/bin/bash


#SBATCH --output=slurm_pyscript_%j_fold4.job
#SBATCH --error=slurm_error_%j_fold4.job
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --nice=10000

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env
python -u train.py --fold 4 > output_fold4_loss.txt
