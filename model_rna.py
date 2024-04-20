import torch
from torch.nn import ModuleList, ReLU, Embedding
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    BatchNorm,
)
from torch_geometric.nn.models import GCN
from einops import rearrange

class RNAGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_out_features, dim=16, n_layers:int=1, device:str='cuda') -> None:
        super().__init__()
        self.device = device
        self.n_layers = n_layers
        self.layers = []
        self.conv_in = GCN(num_node_features, hidden_channels=dim, num_layers=n_layers, out_channels=dim)

        # for _ in range(n_layers):
        #     self.layers.append(
        #         ModuleList(
        #             [
        #                 GATConv(dim, dim).cuda(),
        #                 BatchNorm(dim).cuda(),
        #                 ReLU().cuda()
        #             ]
        #         )
        #     )

        self.conv_out = GATConv(dim, dim)
        self.conv1 = torch.nn.Conv1d(8, 8, 3, padding=1)
        self.mlp_out1 = torch.nn.Linear(dim, num_out_features)
        self.mlp_out2 = torch.nn.Linear(num_out_features, num_out_features)

    def forward(self, data):
        # edge_index_g = data.edge_index[data.edge_attr[:, 1] == 1]
        # edge_index_l = data.edge_index[data.edge_attr[:, 1] == 1]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # create and embedding from the first column of x
        nt_emb = Embedding(4, 32)
        nt_emb.to(self.device)

        nt = nt_emb(x[:, 0].long())
        pos_emb = x[:, 1:]
        # x = torch.cat((nt, pos_emb), dim=1)
        x = nt

        x = self.conv_in(x, edge_index, edge_attr=edge_attr)
        # for gcn, batch_norm, relu in self.layers:
        #     x = gcn(x, edge_index, edge_attr)
        #     x = batch_norm(x)
        #     x = relu(x)
        out = self.conv_out(x, edge_index, edge_attr=edge_attr)
        out = rearrange(out, 'n (p c) -> n p c', p=8)
        out = F.relu(out)
        out = self.conv1(out)
        out = F.relu(out)
        out = rearrange(out, 'n p c -> n (p c)')
        out = self.mlp_out1(out)
        out =  F.relu(out)
        out = self.mlp_out2(out)
        return out
