import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
import wandb

from models import PAMNet, Config
from datasets import RNAPDBDataset
from utils import Sampler
from losses import p_losses

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)


def test(model, loader, device, sampler, args):
    model.eval()
    losses = []
    for data, name in loader:
        data = data.to(device)
        t = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long() # Generate random timesteps
        graphs_t = t[data.batch]
        loss = p_losses(model, data, graphs_t, sampler=sampler, loss_type="huber")
        losses.append(loss.item())
    return np.mean(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-Puzzles', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=64, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=0.03, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=0.200, help='cutoff in global layer')
    parser.add_argument('--timesteps', type=int, default=500, help='timesteps')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    args = parser.parse_args()
    
    if args.wandb:
        wandb.login()
        wandb.init(project='RNA-GNN-Diffusion', config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    # Creat dataset
    path = osp.join('.', 'data', args.dataset)
    train_dataset = RNAPDBDataset(path, name='train-raw-pkl').shuffle()
    val_dataset = RNAPDBDataset(path, name='val-raw-pkl')

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loaded!")
    for data, name in train_loader:
        print(data)
        break

    sampler = Sampler(timesteps=args.timesteps)
    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)

    model = PAMNet(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("Start training!")
    best_val_loss = None
    for epoch in range(args.epochs):
        model.train()
        step = 0
        losses = []
        for data, name in train_loader:
            
            data = data.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long() # Generate random timesteps
            graphs_t = t[data.batch]
            
            loss = p_losses(model, data, graphs_t, sampler=sampler, loss_type="huber")

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 100 == 0 and step != 0:
                print(f"Step: {step}, Loss: {loss.item():.4f}")
            step += 1

        val_loss = test(model, val_loader, device, sampler, args)

        # print('Epoch: {:03d}, Train Loss: {:.7f}, Val Loss: {:.7f}'.format(epoch+1, train_loss, val_loss))
        if args.wandb:
            wandb.log({'Train Loss': np.mean(losses), 'Val Loss': val_loss})
        print(f'Epoch: {epoch+1}, Loss: {np.mean(losses):.4f}, Val Loss: {val_loss:.4f}')
        
        # save_folder = os.path.join(".", "save", args.dataset)
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)

        if epoch %100 == 0:
            torch.save(model.state_dict(), f"./save/model_{epoch}.h5")

        # if best_val_loss is None or val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), os.path.join(save_folder, "best_model.h5"))
    torch.save(model.state_dict(), f"./save/model_{epoch}.h5")

if __name__ == "__main__":
    main()