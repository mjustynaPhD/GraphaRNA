from utils import SamplingMask
from datasets import RNAPDBDataset
from torch_geometric.loader import DataLoader

class TestMasking:
    test_file = "tests/sampling_masks.txt"
    data_path = "data/full_PDB/"

    def test_loading_sampling_resids(self):
        expected_dict = {
            '1ddy_A':['10-13','30-33'],
            '8FZA_1_A':['8-11','27-30']
        }
        sampling_mask = SamplingMask(self.test_file)
        resids = sampling_mask.load_sampling_resids()
        assert resids == expected_dict

    def test_get_sampling_mask(self):
        val_dataset = RNAPDBDataset(self.data_path, name='my-tests', mode='coarse-grain')
        data_loader = DataLoader(val_dataset, batch_size=4)
        sampling_mask = SamplingMask(self.test_file)
        for data, name, seqs in data_loader:
            mask = sampling_mask.get_mask(data, name)
            assert mask is not None
            assert mask.shape[0] == data.x.shape[0]
        
        sampling_mask = SamplingMask(None)
        for data, name, seqs in data_loader:
            mask = sampling_mask.get_mask(data, name)
            assert all(mask == False)
    