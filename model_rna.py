import torch
from torch.nn import ModuleList, ReLU
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    BatchNorm,
)


class RNAGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_out_features, dim=16, n_layers:int=1) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.layers = []
        self.conv_in = GATConv(num_node_features, dim)

        for _ in range(n_layers):
            self.layers.append(
                ModuleList(
                    [
                        GATConv(dim, dim).cuda(),
                        BatchNorm(dim).cuda(),
                        ReLU().cuda()
                    ]
                )
            )

        self.conv_out = GATConv(dim, num_out_features)

    def forward(self, data):
        # edge_index_g = data.edge_index[data.edge_attr[:, 1] == 1]
        # edge_index_l = data.edge_index[data.edge_attr[:, 1] == 1]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv_in(x, edge_index, edge_attr=edge_attr)
        for gcn, batch_norm, relu in self.layers:
            x = gcn(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = relu(x)
        out = self.conv_out(x, edge_index, edge_attr=edge_attr)
        return out
