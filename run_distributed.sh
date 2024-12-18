#!/usr/bin/bash -i
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p hgx
#SBATCH --gres=gpu:04
#SBATCH -t 168:00:00

source ~/.bashrc
conda activate gnn

torchrun \
    --standalone \
    --nproc_per_node=4 \
    main_rna_pdb.py --dataset rna3db --epoch=10001 --batch_size=4 --dim=256 --n_layer=4 --lr=1e-4 --timesteps=1000 --mode=coarse-grain --knn=5 --wandb --lr-step=100000 --blocks=4 --load
