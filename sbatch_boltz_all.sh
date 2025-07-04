#!/bin/bash
#SBATCH -p hgx
#SBATCH --gres=gpu:01
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=64:00:00


source ~/.bashrc
conda activate boltz

task_id=${SLURM_ARRAY_TASK_ID:-0}

python run_boltz_parallel.py $task_id 10