#!/bin/bash
#SBATCH -p hgx
#SBATCH --gres=gpu:01
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=64:00:00


source ~/.bashrc
conda activate gnn_test

task_id=${SLURM_ARRAY_TASK_ID:-0}

python src/grapharna/sample_all_rna.py --n_layer 1 --blocks 12 --eval_batch_num 8 --eval_batch_idx $task_id