import os
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
import wandb

from rnabbit.utils import MCQStructure, Evaluate3D

from model_rna import RNAGNN
from datasets import TorsionAnglesDataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(model, loader, device):
    model.eval()

    pred_list = []
    y_list = []

    for batch in loader:
        data, name = batch
        data = data.to(device)
        pred = model(data)
        pred_list += pred.reshape(-1).tolist()
        y_list += data.y.reshape(-1).tolist()

    pred = np.array(pred_list).reshape(-1,)
    pred = torch.tensor(pred).to(device)

    y = np.array(y_list).reshape(-1,)
    y = torch.tensor(y).to(device)

    loss = F.smooth_l1_loss(pred, y)
    return loss.item(), np.array(pred_list).reshape(-1,)

def evaluate_samples(samples, seq, evaluator, samples_path, epoch, nan_eps, name, mode='tor_ang', prefix='e'):
    rmsds = []
    for i, ss in enumerate(zip(samples, seq)):
        structs, seq = ss
        seq = seq.squeeze().cpu().detach().numpy()
        structs = structs.reshape((2, 17, -1))
        structure = MCQStructure(structs, mode=mode, angle_eps=nan_eps, seq=seq)
        structure.save(samples_path, f"{prefix}_{epoch}_{name[i].replace('.pkl', '')}")
        rmsd = evaluator.evaluate_rmsd(f"{prefix}_{epoch}_{name[i].replace('.pkl', '')}")
        rmsds.append(rmsd)
    return rmsds

def predict_samples(model, loader, device, evaluator, samples_path, epoch, nan_eps, mode='tor_ang', prefix='e'):
    model.eval()
    
    preds = []
    seqs = []
    names = []

    for batch in loader:
        data, name = batch
        data = data.to(device)
        pred = model(data)
        preds.append(pred.cpu().detach().numpy())
        # seq = data.x
        # numerical_seq = np.argmax(seq.cpu().detach(), axis=1)
        seqs.append(data.x[:, 0] * 4)
        names.append(name[0])

    
    rmsds = evaluate_samples(preds, seqs, evaluator, samples_path, epoch, nan_eps, names, mode=mode, prefix=prefix)
    return rmsds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-Puzzles', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=16, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--sample_freq', type=int, default=1, help='Sample frequency for logging')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    else:
        raise Exception("No GPU available!")

    
    set_seed(args.seed)

    # Creat dataset
    path = osp.join('.', 'data', args.dataset)
    train_dataset = TorsionAnglesDataset(path, name='train', use_node_attr=True)
    val_dataset = TorsionAnglesDataset(path, name='val', use_node_attr=True)

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    sample_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    evaluator = Evaluate3D(reference_pdbs_path='/data/3d/bgsu_pdbs/', artifacts_path='artifacts/')
    print("Data loaded!")
    # for data in train_loader:
    #     print(data)
    #     break
    
    
    wandb.login()
    wandb.init(
        project="RNA-Torsion",
        config=args,
        mode="online" if args.wandb else "disabled"
        )
    

    model = RNAGNN(33, 34, dim=args.dim, n_layers=args.n_layer)
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    

    print("Start training!")
    best_val_loss = None
    for epoch in range(args.epochs):
        model.train()

        for step, batch in enumerate(train_loader):
            data, name = batch
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.smooth_l1_loss(output, data.y)
            # loss = F.mse_loss(output, data.y)
            loss.backward()
            optimizer.step()
        
        train_loss, _ = test(model, train_loader, device)
        val_loss, _ = test(model, val_loader, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        print('Epoch: {:03d}, Train Loss: {:.7f}, Val Loss: {:.7f}'.format(epoch+1, train_loss, val_loss))

        if epoch % args.sample_freq == 0:
            samples_path = os.path.join(".", "artifacts")
            if not os.path.exists(samples_path):
                os.makedirs(samples_path)
            rmsds = predict_samples(model, sample_loader, device, evaluator, samples_path, epoch, 1e-3, mode='tor_ang', prefix='e')
            print(f"Epoch: {epoch}, RMSD: {np.mean(rmsds)}, Min RMSD: {np.min(rmsds)}")
            wandb.log({"rmsd": np.mean(rmsds), "min_rmsd": np.min(rmsds), "std_rmsd": np.std(rmsds), "max_rmsd": np.max(rmsds)})

        
        save_folder = os.path.join(".", "save", args.dataset)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_folder, "best_model.h5"))



if __name__ == "__main__":
    main()