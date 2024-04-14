import torch
from torch.nn import ModuleList, ReLU
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    BatchNorm
)

class RNAGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_out_features, dim=16, n_layers:int=1) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.layers = []
        self.conv_in = GCNConv(num_node_features, dim)

        for _ in range(n_layers):
            self.layers.append(
                ModuleList(
                    [
                        GCNConv(dim, dim).cuda(),
                        BatchNorm(dim).cuda(),
                        ReLU().cuda()
                    ]
                )
            )
            
        self.conv_sin = GCNConv(dim, num_out_features)
        self.conv_cos = GCNConv(dim, num_out_features)

    def forward(self, data):
        # edge_index_g = data.edge_index[data.edge_attr[:, 1] == 1]
        # edge_index_l = data.edge_index[data.edge_attr[:, 1] == 1]
        x, edge_index = data.x, data.edge_index
        x = self.conv_in(x, edge_index)
        for gcn, batch_norm, relu in self.layers:
            x = gcn(x, edge_index)
            x = batch_norm(x)
            x = relu(x)
        sin_out = self.conv_sin(x, edge_index)
        cos_out = self.conv_cos(x, edge_index)
        return F.relu(sin_out), F.relu(cos_out)
        

        