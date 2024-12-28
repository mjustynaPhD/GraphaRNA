import os
import torch
import numpy as np
import pickle
from torch_geometric.data import Data, Dataset
from constants import BACKBONE_ATOMS, REV_RESIDUES

class RNAPDBDataset(Dataset):
    def __init__(self,
                 path: str,
                 name: str,
                 file_extension: str='.pkl',
                 mode: str='backbone'
                 ):
        super(RNAPDBDataset, self).__init__(path)
        self.path = os.path.join(path, name)
        self.files = sorted(os.listdir(self.path))
        self.files = os.listdir(self.path)
        self.files = [f for f in self.files if f.endswith(file_extension)]
        self.to_tensor = torch.tensor
        if mode not in ['backbone', 'all', 'coarse-grain', 'p_only']:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def len(self):
        return len(self.files)

    def get(self, idx):
        data_x, edges, name, edges_type, seq = self.get_raw_sample(idx)
        data = Data(
            x=data_x,
            edge_index=edges.t().contiguous(),
            edge_attr=edges_type
        )
        return data, name, "".join(seq)

    def get_raw_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.files[idx]
        path = os.path.join(self.path, file)
        sample = self.load_pickle(path)
        atoms_types = self.to_tensor(sample['atoms']).unsqueeze(1).float()
        atoms_pos = self.to_tensor(sample['pos']).float()
        atoms_pos_mean = atoms_pos[sample['coords_updated']].mean(dim=0)
        atoms_pos[sample['coords_updated']] -= atoms_pos_mean # Center around point (0,0,0)
        # atoms_pos /= 10
        c2 = c4_or_c6 = n1_or_n9 = None
        if self.mode == 'backbone':
            atoms_pos, atoms_types, c4_primes, residues = self.backbone_only(atoms_pos, atoms_types, sample)
        elif self.mode == 'coarse-grain':
            atoms_pos, atoms_types, c4_primes, residues, c2, c4_or_c6, n1_or_n9 = self.coarse_grain(atoms_pos, atoms_types, sample)
        elif self.mode == 'all':
            c4_primes = sample['c4_primes']
            residues = sample['residues']
            c2 = sample['c2']
            c4_or_c6 = sample['c4_or_c6']
            n1_or_n9 = sample['n1_or_n9']
        elif self.mode == 'p_only':
            atoms_pos, atoms_types, residues = self.p_only(atoms_pos, atoms_types, sample)

        name = sample['name'].replace('.pkl', '')
        residue_names = np.array([REV_RESIDUES[res] for res in residues])
        residues = torch.nn.functional.one_hot(torch.tensor(residues).to(torch.int64), num_classes=4).float()
        
        if self.mode in ['coarse-grain', 'all', 'backbone']:
            data_x, residue_names = self.get_data_x_cg(atoms_pos, sample, c4_primes, c2, c4_or_c6, n1_or_n9, residue_names, residues, atoms_types)
        elif self.mode == 'p_only':
            coords_mask = torch.tensor(sample['coords_updated']).bool().unsqueeze(1)
            data_x = torch.cat((atoms_pos, atoms_types, residues, coords_mask), dim=1)

        edges = torch.tensor(sample['edges'])
        if len(edges.shape) == 3:
            edges = edges.squeeze(2)
        
        return data_x,\
                edges,\
                name,\
                torch.nn.functional.one_hot(torch.tensor(sample['edge_type']).to(torch.int64), num_classes=3).float(),\
                residue_names

    def get_data_x_cg(self, atoms_pos, sample, c4_primes, c2, c4_or_c6, n1_or_n9, residue_names, residues, atoms_types):
        
        # convert atom_types to one-hot encoding (C, O, N, P)
        atoms_types = torch.nn.functional.one_hot(atoms_types.to(torch.int64), num_classes=4).float()
        atoms_types = atoms_types.squeeze(1)
        coords_mask = torch.tensor(sample['coords_updated']).bool().unsqueeze(1)
        c4_primes = torch.tensor(c4_primes).float().unsqueeze(1)
        if c2 is not None:
            c2 = torch.tensor(c2).float().unsqueeze(1)
            c4_or_c6 = torch.tensor(c4_or_c6).float().unsqueeze(1)
            n1_or_n9 = torch.tensor(n1_or_n9).float().unsqueeze(1)
        
        if n1_or_n9 is not None:
            n_pos = torch.where(n1_or_n9)[0]
            residue_names = residue_names[n_pos]
        

        if c2 is not None:
            data_x = torch.cat((atoms_pos, atoms_types, residues, c4_primes, c2, c4_or_c6, n1_or_n9, coords_mask), dim=1)
        elif c4_primes is not None:
            data_x = torch.cat((atoms_pos, atoms_types, residues, c4_primes, coords_mask), dim=1)
        return data_x, residue_names

    def backbone_only(self, atom_pos, atom_types, sample):
        mask = [True if atom in BACKBONE_ATOMS else False for atom in sample['symbols']]
        c4_primes = sample['c4_primes']
        residues = sample['residues']
        return atom_pos[mask], atom_types[mask], c4_primes[mask], residues[mask]
    
    def coarse_grain(self, atom_pos, atom_types, sample):
        # mask = sample['crs-grain-mask']
        if 'c4_primes' in sample:
            c4_primes = sample['c4_primes']
            c2 = sample['c2']
            c4_or_c6 = sample['c4_or_c6']
            n1_or_n9 = sample['n1_or_n9']
        else:
            c4_primes = None
            c2 = None
            c4_or_c6 = None
            n1_or_n9 = None
        residues = sample['residues']
        return atom_pos, atom_types, c4_primes, residues, c2, c4_or_c6, n1_or_n9

    def p_only(self, atom_pos, atom_types, sample):
        residues = sample['residues']
        return atom_pos, atom_types, residues


    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return self.files

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)