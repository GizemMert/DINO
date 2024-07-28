#!/bin/bash


#SBATCH -o slurm_pyscript_%j_fold3.job
#SBATCH -e slurm_error_%j_fold3.job
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH --nice=10000


python -u train.py --fold 3 > output_fold3_loss.txt
