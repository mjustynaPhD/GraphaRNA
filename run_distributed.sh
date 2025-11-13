#!/usr/bin/bash -i
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p hgx
#SBATCH --gres=gpu:2
#SBATCH -t 168:00:00


source .venv/bin/activate
# torchrun \
#     --standalone \
#     --nproc_per_node=6 \
#     src/grapharna/main_rna_pdb.py --dataset RNA-PDB-clean --epoch=1000 --batch_size=16 --dim=256 --n_layer=6 --lr=1e-3 --timesteps=5000 --cutoff_l=0.5 --cutoff_g=1.6 --mode=coarse-grain --knn=20 --wandb --lr-step=30 --blocks=6

torchrun \
    --standalone \
    --nproc_per_node=2 \
    src/grapharna/main_rna_pdb.py --dataset P-only --epoch=1000 --batch_size=2 --dim=256 --n_layer=2 --lr=1e-3 --timesteps=2000 --cutoff_l=0.5 --cutoff_g=1.6 --mode=p-only --knn=5 --lr-step=30 --blocks=6 --wandb