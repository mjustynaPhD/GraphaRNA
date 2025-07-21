import os
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
import wandb

from grapharna.models import PAMNet, Config
from grapharna.datasets import RNAPDBDataset
from grapharna.utils import Sampler, SampleToPDB, SamplingMask
from grapharna.losses import p_losses

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup():
    dist.destroy_process_group()

def validation(model, loader, device, sampler, args):
    model.eval()
    losses = []
    denoise_losses = []
    with torch.no_grad():
        for data, name, seqs in loader:
            mask = data.x[:, -1].bool()
            data = data.to(device)
            t = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long() # Generate random timesteps
            graphs_t = t[data.batch]
            loss, denoise_loss = p_losses(model, data, seqs, graphs_t, sampler=sampler, loss_type="huber", mask=mask)
            losses.append(loss.item())
            denoise_losses.append(denoise_loss.item())
    model.train()
    return np.mean(losses), np.mean(denoise_losses)

def sample(model, loader, device, sampler, epoch, args, num_batches=None, exp_name: str = "run", ):
    model.eval()
    s = SampleToPDB()
    # mask_sampler = SamplingMask(args.sampling_resids, device=device)
    s_counter = 0

    with torch.no_grad():
        for data, name, seqs in loader:
            print(f"Sample batch {s_counter}")
            data = data.to(device)
            # sampling_mask = mask_sampler.get_mask(data, name)
            samples = sampler.sample(model, seqs, data)[-1]
            s.to('pdb', samples, f"./samples/{exp_name}/{epoch}", name)
            # s.to('xyz', samples, f"./samples/{exp_name}/{epoch}", name)
            # s.to('trafl', samples, f"./samples/{exp_name}/{epoch}", name)
            s_counter += 1

            if num_batches is not None and s_counter >= num_batches:
                break

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(world_size):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-Puzzles', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=64, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=5, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=16, help='cutoff in global layer')
    parser.add_argument('--timesteps', type=int, default=500, help='timesteps')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--mode', type=str, default='coarse-grain', help='Mode of the dataset')
    parser.add_argument('--lr-step', type=int, default=30, help='Step size for learning rate scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.9, help='Gamma for learning rate scheduler')
    parser.add_argument('--knns', type=int, default=2, help='Number of knn neighbors')
    parser.add_argument('--blocks', type=int, default=4, help='Number of transformer blocks in the model')
    parser.add_argument('--load', action='store_true', help='Path to the model to load')
    args = parser.parse_args()
    
    # setup(rank, world_size)
    dist.init_process_group("nccl")
    rank = int(os.environ['LOCAL_RANK'])

    if args.wandb and rank == 0:
        wandb.login()
        run = wandb.init(project='RNA-GNN-ablation', config=args)
        exp_name = run.name
    else:
        exp_name = "test"

    device = rank
    set_seed(args.seed)
    print(f"Rank: {rank} Device:{device}")

    # Creat dataset
    path = osp.join('.', 'data', args.dataset)
    train_dataset = RNAPDBDataset(path, name='train-pkl', mode=args.mode).shuffle()
    val_dataset = RNAPDBDataset(path, name='val-pkl', mode=args.mode)
   
    dist_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dist_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=dist_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_dist_sampler)
    # samp_loader = DataLoader(samp_dataset, batch_size=6, shuffle=False)
    print("Data loaded!")
    for data, name, seqs in train_loader:
        print(data)
        break

    sampler = Sampler(timesteps=args.timesteps)
    config = Config(dataset=args.dataset,
                    dim=args.dim,
                    n_layer=args.n_layer,
                    cutoff_l=args.cutoff_l,
                    cutoff_g=args.cutoff_g,
                    mode=args.mode,
                    knns=args.knns,
                    transformer_blocks=args.blocks
                    )

    model = PAMNet(config).to(device)
    # load state dict of a pre-trained model
    if args.load:
        model.load_state_dict(torch.load("save/ruby-mountain-45/model_400.h5"))

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    print("Start training!")
    torch.autograd.set_detect_anomaly(True)
    
    dist.barrier()
    for epoch in range(args.epochs):
        model.train()
        step = 0
        losses = []
        denoise_losses = []
        for data, name, seqs in train_loader:
            mask = data.x[:, -1].bool()
            data = data.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long() # Generate random timesteps
            graphs_t = t[data.batch]
            
            loss_all, loss_denoise = p_losses(model, data, seqs, graphs_t, sampler=sampler, loss_type="huber", mask=mask)

            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradients
            optimizer.step()
            losses.append(loss_all.item())
            denoise_losses.append(loss_denoise.item())
            if step % 5 == 0 and step != 0:
                val_loss, val_denoise_loss = validation(model, val_loader, device, sampler, args)
                if args.wandb and rank == 0:
                    print(f'Epoch: {epoch+1}, Step: {step}, Loss: {np.mean(losses):.4f}, Denoise Loss: {np.mean(denoise_losses):.4f}, Val Loss: {val_loss:.4f}, Val Denoise Loss: {val_denoise_loss:.4f} LR: {scheduler.get_last_lr()[0]}')
                    wandb.log({'Train Loss': np.mean(losses), 'Val Loss': val_loss, 'Denoise Loss': np.mean(denoise_losses), 'Val Denoise Loss': val_denoise_loss, "LR": scheduler.get_last_lr()[0]})
                losses = []
                denoise_losses = []
            elif not args.wandb and rank == 0:
                print(f"Epoch: {epoch}, step: {step}, loss: {loss_all.item():.4f} ")
            step += 1
        scheduler.step()
        
        if args.wandb and rank == 0 and losses:
            wandb.log({'Train Loss': np.mean(losses), 'Val Loss': val_loss, 'Denoise Loss': np.mean(denoise_losses), 'Val Denoise Loss': val_denoise_loss, "LR": scheduler.get_last_lr()[0]})
        if rank == 0:
            print(f'Epoch: {epoch+1}, Loss: {np.mean(losses):.4f}, Denoise Loss: {np.mean(denoise_losses):.4f}, Val Loss: {val_loss:.4f}, Val Denoise Loss: {val_denoise_loss:.4f}, LR: {scheduler.get_last_lr()[0]}')
        

        
        save_folder = f"./save/{exp_name}"
        if not os.path.exists(save_folder) and rank==0:
            os.makedirs(save_folder)

        if epoch % 10 == 0 and epoch > 0 and rank==0:
            print(f"Saving model at epoch {epoch} to {save_folder}")
            torch.save(model.module.state_dict(), f"{save_folder}/model_{epoch}.h5")

    if rank == 0:
        torch.save(model.module.state_dict(), f"{save_folder}/model_{epoch}.h5")
    
    dist.destroy_process_group()


def run(main_fn, world_size):
    print("Running DDP with world size: ", world_size)
    mp.spawn(main_fn,
             args=(world_size, ),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    main(world_size=world_size)
