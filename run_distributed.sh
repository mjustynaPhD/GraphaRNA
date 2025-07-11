#!/usr/bin/bash -i
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p proxima
#SBATCH --gres=gpu:04
#SBATCH -t 168:00:00

source ~/.bashrc
conda activate gnn_test

torchrun \
    --standalone \
    --nproc_per_node=4 \
    src/grapharna/main_rna_pdb.py --dataset RNA-PDB-clean --epoch=801 --batch_size=16 --dim=256 --n_layer=6 --lr=1e-4 --timesteps=5000 --cutoff_l=0.5 --cutoff_g=1.6 --mode=coarse-grain --knn=20 --lr-step=50 --blocks=6 --wandb 
