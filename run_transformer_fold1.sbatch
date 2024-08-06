#!/bin/bash
#SBATCH --output=slurm_pyscript_%j_fold1.job
#SBATCH --error=slurm_error_%j_fold1.job
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --nice=10000

export TMPDIR=/tmp

ulimit -n 409

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env
python -u train.py --fold 1 > output_fold1_loss.txt