import os
import torch
import torch.distributed as dist

def main():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    print(f"rank={dist.get_rank()} local={os.environ.get('LOCAL_RANK')} world={dist.get_world_size()}")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()