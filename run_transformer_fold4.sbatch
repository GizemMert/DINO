#!/bin/bash


#SBATCH -o slurm_pyscript_%j_fold4.job
#SBATCH -e slurm_error_%j_fold4.job
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH --nice=10000

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env
python -u train.py --fold 4 > output_fold4_loss.txt
