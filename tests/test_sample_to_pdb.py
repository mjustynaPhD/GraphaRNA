import pytest
from torch.utils.data import DataLoader
from datasets import RNAPDBDataset
from utils import SampleToPDB

class TestSampleToPDB:
    data_path = "data/7QR4/"
    out_path = "tests/test_output/"

    def test_for_P_only(self):
        p_data_path = "tests/"
        val_ds = RNAPDBDataset(p_data_path, name='test_data', mode='p_only')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name, _seq in val_loader:
            sample.to("pdb", data, self.out_path, name)
            break

    def test_to_pdb(self):
        # Test the to_pdb method
        val_ds = RNAPDBDataset(self.data_path, name='test-pkl', mode='coarse-grain')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name, _seq in val_loader:
            sample.to("pdb", data, self.out_path, name)
            break
        # Add assertions to verify the output

    def test_write_xyz_all(self):
        val_ds = RNAPDBDataset(self.data_path, name='val-raw-pkl', mode='all')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name in val_loader:
            sample.to("xyz", data, self.out_path, name, post_fix='_all')
            break

    def test_write_xyz_backbone(self):
        val_ds = RNAPDBDataset(self.data_path, name='val-raw-pkl', mode='backbone')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name in val_loader:
            sample.to("xyz", data, self.out_path, name, post_fix='_bb')
            break

    def test_write_xyz_coarse(self):
        val_ds = RNAPDBDataset(self.data_path, name='val-raw-pkl', mode='coarse-grain')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name in val_loader:
            sample.to("xyz", data, self.out_path, name, post_fix='_cgr')
            break
    
    def test_write_trafl(self):
        val_ds = RNAPDBDataset(self.data_path, name='val-raw-pkl', mode='coarse-grain')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name in val_loader:
            sample.to("trafl", data, self.out_path, name)
            break
