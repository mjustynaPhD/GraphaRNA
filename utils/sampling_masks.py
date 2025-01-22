import torch

class SamplingMask():
    def __init__(self, path: str = None, device='cpu'):
        self.sampling_resids = path
        self.resids_dict = self.load_sampling_resids()
        self.device = device
        # self.fixed_ps = args.fixed_ps

    def load_sampling_resids(self):
        if self.sampling_resids is None:
            return {}
        else:
            print("Loading sampling residues...")
            resids = self.read_sampling_resids()
            return resids

    def read_sampling_resids(self):
        with open(self.sampling_resids, 'r') as f:
                sampling_resids = f.readlines()
        samplings = {}
        last_file = ""
        for s in sampling_resids:
            if s.startswith(">"):
                last_file = s[1:].strip()
            elif last_file and s != '\n':
                samplings[last_file] = s.strip().split(';')
        return samplings

    def parse_range(self, range_str):
        if '-' not in range_str:
            return int(range_str), int(range_str)+1
        start, end = range_str.split('-')
        return int(start)-1, int(end)-1

    def get_mask(self, data, name):
        names = [n.split('.')[0] for n in name]
        names_masks = [n in self.resids_dict for n in names]
        # find indeces where mask is True
        indices = [i for i, x in enumerate(names_masks) if x]
        mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        mask = mask.to(self.device)
        for index in indices:
            mask = self.create_mask(data, index, mask, names[index])
        return mask

    def create_mask(self, data, index, mask, name):
        batch_indeces = torch.where(data.batch == index)[0]
        assert batch_indeces.shape[0] % 5 == 0
        resids = self.resids_dict[name]
        batch_indeces.to(self.device)
        interest_mask = mask[batch_indeces]
        interest_mask = interest_mask.reshape(-1, 5)
        for r in resids:
            start, end = self.parse_range(r)
            interest_mask[start:end] = True
        mask[batch_indeces] = interest_mask.reshape(-1)
        return mask
