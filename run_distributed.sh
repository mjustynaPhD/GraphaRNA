#!/usr/bin/bash -i
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --exclude=gpu73
#SBATCH --nodes=8
#SBATCH -c 8
#SBATCH -p proxima
#SBATCH --gres=gpu:04
#SBATCH -t 168:00:00

cd ~/pl0631-01/scratch/mjust/GraphaRNA/
source .venv/bin/activate

MASTER_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=$(getent hosts $MASTER_HOSTNAME | awk '{ print $1 }')
MASTER_PORT=29500


echo "SLURM_NODEID=$SLURM_NODEID"
echo "MASETER_HOSTNAME=$MASTER_HOSTNAME"
echo "MASTER_ADDR=$MASTER_ADDR"

srun \
    uv run torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/grapharna/main_rna_pdb.py --dataset full-3d --epoch=1000 --batch_size=1 --dim=128 --n_layer=4 --lr=1e-3 --timesteps=2000 --cutoff_l=0.5 --cutoff_g=1.6 --mode=coarse-grain --knn=10 --wandb --lr-step=30 --blocks=6
