import os
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader

from model_rna import RNAGNN
from datasets import TorsionAnglesDataset

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(model, loader, device):
    model.eval()

    sin_pred_list = []
    cos_pred_list = []
    sin_y_list = []
    cos_y_list = []

    for batch in loader:
        data, name = batch
        data = data.to(device)
        sin_pred, cos_pred = model(data)
        
        sin_pred_list += sin_pred.reshape(-1).tolist()
        sin_y_list += data.y[:, 0, :].reshape(-1).tolist()
        cos_pred_list += cos_pred.reshape(-1).tolist()
        cos_y_list += data.y[:, 1, :].reshape(-1).tolist()


    sin_pred = np.array(sin_pred_list).reshape(-1,)
    sin_pred = torch.tensor(sin_pred).to(device)
    cos_pred = np.array(cos_pred_list).reshape(-1,)
    cos_pred = torch.tensor(cos_pred).to(device)

    sin_y = np.array(sin_y_list).reshape(-1,)
    sin_y = torch.tensor(sin_y).to(device)
    cos_y = np.array(cos_y_list).reshape(-1,)
    cos_y = torch.tensor(cos_y).to(device)

    sin_loss = F.smooth_l1_loss(sin_pred, sin_y)
    cos_loss = F.smooth_l1_loss(cos_pred, cos_y)
    return sin_loss.item()+cos_loss.item(), np.array(sin_pred_list).reshape(-1,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-Puzzles', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=64, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=2.6, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=20.0, help='cutoff in global layer')
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
    print("Data loaded!")
    # for data in train_loader:
    #     print(data)
    #     break

    # config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)
    # model = PAMNet(config).to(device)
    model = RNAGNN(1, 17, n_layers=1)
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

            sin_out, cos_out = model(data)
            sin_loss = F.smooth_l1_loss(sin_out, data.y[:, 0, :])
            cos_loss = F.smooth_l1_loss(cos_out, data.y[:, 1, :])
            loss = sin_loss + cos_loss
            loss.backward()
            optimizer.step()
        
        train_loss, _ = test(model, train_loader, device)
        val_loss, _ = test(model, val_loader, device)

        print('Epoch: {:03d}, Train Loss: {:.7f}, Val Loss: {:.7f}'.format(epoch+1, train_loss, val_loss))
        
        save_folder = os.path.join(".", "save", args.dataset)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_folder, "best_model.h5"))


if __name__ == "__main__":
    main()