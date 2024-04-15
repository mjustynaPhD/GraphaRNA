import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

class TorsionAnglesDataset(Dataset):
    encode_2d = {
        '(' : 1,
        ')' : 2,
        '.' : 3,
        }

    # alphabet = {
    #     'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 5
    # }
    
    def __init__(self,
                 path: str,
                 name: str,
                 transform=None,
                 file_extension: str = "pkl",
                 use_edge_attr: bool = True,
                 use_node_attr: bool = True,
                 pre_transform=None,
                 pre_filter=None
                 ):
        super(TorsionAnglesDataset, self).__init__(path, transform, pre_transform, pre_filter)
        self.path = os.path.join(path, name)
        self.transform = transform
        self.files = sorted(os.listdir(self.path))
        self.files = os.listdir(self.path)
        self.files = [f for f in self.files if f.endswith(file_extension)]
        self.use_edge_attr = use_edge_attr
        self.use_node_attr = use_node_attr
        self.to_tensor = torch.tensor

    def len(self):
        return len(self.files)

    def get(self, idx):
        node_attr, edge_indeces, edge_attr, torsions = self.get_raw_sample(idx)
        if self.transform:
            torsions = self.transform(torsions)
        name = self.files[idx]
        data = Data(
            x=node_attr,
            edge_index=edge_indeces,
            edge_attr=edge_attr,
            y=torsions.float(),
        )
        return data, name

    def get_raw_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.files[idx]
        path = os.path.join(self.path, file)
        sample = self.load_pickle(path)

        seq = self.to_tensor(sample['seq_num'])
        seq_graph = self.sequence_to_graph(seq)
        # sequence to categorical
        # node_attr = self.seq_one_hot(seq)
        node_attr = seq.unsqueeze(1).float()
        torsions = np.array(sample['tor_ang']).astype(dtype=np.float16)
        torsions = self.to_tensor(self.encode_torsions(torsions))
        torsions = torsions.permute(2, 0, 1)
        torsions = torsions[:len(seq)]
        torsions = torsions.reshape(torsions.shape[0], -1)
        # ss = sample['pred_2d']
        ss = sample['dot_2d']
        ss = self.to_tensor([self.encode_2d[c] for c in ss])
        ss_graph = self.ss_to_pairs(ss)
        if self.use_edge_attr:
            distances = sample['distances']
            distances = self.to_tensor(distances)
        else:
            distances = None
        seq_edge_attr = self.get_edge_attr(seq_graph, distances)
        seq_edge_attr = torch.cat((seq_edge_attr, torch.zeros_like(seq_edge_attr)), dim=1)
        ss_edge_attr = self.get_edge_attr(ss_graph, distances)
        ss_edge_attr = torch.cat((ss_edge_attr, torch.ones_like(ss_edge_attr)), dim=1)
        edge_indeces = torch.cat((seq_graph, ss_graph), dim=1)
        edge_attr = torch.cat((seq_edge_attr, ss_edge_attr), dim=0)
        return node_attr, edge_indeces, edge_attr, torsions

    def encode_torsions(self, degrees):
        radians = np.deg2rad(degrees)
        sin_values = np.sin(radians)
        cos_values = np.cos(radians)
        encoded_values = np.stack((sin_values, cos_values), axis=0)
        return encoded_values
    
    def seq_one_hot(self, seq):
        seq = seq.squeeze()
        seq = seq.numpy()
        seq = np.eye(4)[seq]
        return torch.tensor(seq, dtype=torch.float32)

    def ss_to_pairs(self, ss):
        openings = torch.where(ss == 1)[0].unsqueeze(0)
        closings = reversed(torch.where(ss == 2)[0]).unsqueeze(0)
        pairs = torch.cat((openings, closings), dim=0)
        pairs_rev = torch.cat((closings, openings), dim=0)
        pairs = torch.cat((pairs, pairs_rev), dim=1)
        return pairs

    def get_edge_attr(self, edge_index, distances):
        edge_attr = torch.zeros(edge_index.shape[1], 1)
        for i, (start, end) in enumerate(edge_index.t()):
            edge_attr[i] = distances[start, end]
        return edge_attr

    def sequence_to_graph(self, seq):
        num_nodes = len(seq)
        edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)], dtype=torch.long)
        edge_index = torch.cat((edge_index, torch.flip(edge_index, [1])), dim=0)
        return edge_index.t().contiguous()
    
    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return self.files
    
    @staticmethod
    def pad_sequence(seq, seq_len):
        return torch.nn.functional.pad(
            seq,
            (0, seq_len - seq.shape[0]),
            'constant', value=0
            )

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)