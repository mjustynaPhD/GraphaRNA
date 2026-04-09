#!/bin/bash
#SBATCH --job-name=torch_multi_test
#SBATCH --nodes=2
#SBATCH -p proxima
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

cd ~/pl0631-01/scratch/mjust/GraphaRNA/
source .venv/bin/activate

###############################
# IPv4 only
###############################

###############################
# Choose interface manually
###############################
# Replace eth0 with the correct interface if needed
###############################
# Fixed master = first node
###############################
MASTER_HOST=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
RDZV_ENDPOINT="$MASTER_HOST:29500"

echo "MASTER_ADDR=$MASTER_HOST"
echo "RDZV_ENDPOINT=$RDZV_ENDPOINT"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NODEID=$SLURM_NODEID"

###############################
# Run torchrun
###############################
srun uv run torchrun \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_backend=c10d \
  --rdzv_id=456 \
  --rdzv_endpoint=$RDZV_ENDPOINT \
  test.py
