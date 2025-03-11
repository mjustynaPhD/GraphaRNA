#!/usr/bin/bash -i
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p hgx
#SBATCH --gres=gpu:08
#SBATCH -t 168:00:00

source ~/.bashrc
conda activate gnn

torchrun \
    --standalone \
    --nproc_per_node=8 \
    main_rna_pdb.py --dataset rna3db --epoch=10001 --batch_size=4 --dim=64 --n_layer=2 --lr=1e-3 --timesteps=1000 --mode=coarse-grain --knn=10 --wandb --lr-step=160 --blocks=2
