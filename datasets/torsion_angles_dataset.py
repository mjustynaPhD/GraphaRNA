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
        torsions, seq, ss, distances = self.get_raw_sample(idx)
        if self.transform:
            torsions = self.transform(torsions)
        name = self.files[idx]
        data = Data(
            x=distances.float(),
            seq=seq.float(),
            ss=ss.float(),
            y=torsions.float(),
            name=name
        )
        return data

    def get_raw_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.files[idx]
        path = os.path.join(self.path, file)
        sample = self.load_pickle(path)

        seq = self.to_tensor(sample['seq_num'])
        torsions = np.array(sample['tor_ang']).astype(dtype=np.float16)
        torsions = self.to_tensor(self.encode_torsions(torsions))
        torsions = torsions.permute(2, 0, 1)
        torsions = torsions[:len(seq)]
        # ss = sample['pred_2d']
        ss = sample['dot_2d']
        ss = self.to_tensor([self.encode_2d[c] for c in ss])
        ss_pairs = self.ss_to_pairs(ss)
        if self.use_edge_attr:
            distances = sample['distances']
            distances = distances[:len(seq), :len(seq)]
            distances = self.to_tensor(distances)
        else:
            distances = None
        return torsions, seq, ss_pairs, distances

    def encode_torsions(self, degrees):
        radians = np.deg2rad(degrees)
        sin_values = np.sin(radians)
        cos_values = np.cos(radians)
        encoded_values = np.stack((sin_values, cos_values), axis=0)
        return encoded_values
    
    def ss_to_pairs(self, ss):
        openings = torch.where(ss == 1)[0].unsqueeze(0)
        closings = reversed(torch.where(ss == 2)[0]).unsqueeze(0)
        pairs = torch.cat((openings, closings), dim=0).t()
        return pairs
    
    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return self.files

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)