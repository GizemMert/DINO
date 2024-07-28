#!/bin/bash


#SBATCH -o slurm_error_%j.job
#SBATCH -e slurm_error_%j.job
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH --nice=10000


python -u test.py > output_all_data_fold4.txt
