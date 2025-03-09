import argparse
import torch
from torch_geometric.loader import DataLoader
from models import PAMNet, Config
from datasets import RNAPDBDataset
from utils import Sampler
from main_rna_pdb import sample

from torch_geometric import seed_everything
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-PDB', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=64, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=5, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=16, help='cutoff in global layer')
    parser.add_argument('--timesteps', type=int, default=4000, help='timesteps')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--mode', type=str, default='coarse-grain', help='Mode of the dataset')
    parser.add_argument('--knns', type=int, default=5, help='Number of knns')
    parser.add_argument('--blocks', type=int, default=4, help='Number of transformer blocks')
    parser.add_argument('--sampling-resids', type=str, default=None, help='Residues that will be sampled, while the rest of the structure will remain fixed')
    # parser.add_argument('--fixed-ps', action='store_true', help='If True, P atoms will be fixed and the rest of the structure will be generated. Otherwise, the whole structure will be generated')
    args = parser.parse_args()

    print('Seed:', args.seed)
    set_seed(args.seed)
    # Load the model
    exp_name = "twilight-shadow-129"
    epoch = 900
    model_path = f"save/{exp_name}/model_{epoch}.h5"
    config = Config(dataset=args.dataset,
                    dim=args.dim,
                    n_layer=args.n_layer,
                    cutoff_l=args.cutoff_l,
                    cutoff_g=args.cutoff_g,
                    mode=args.mode,
                    knns=args.knns,
                    transformer_blocks=args.blocks
                    )
    model = PAMNet(config)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded!")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    model.to(device)
    dataset = 'my-tests' # 'test-pkl' #'my-tests'
    dataset = 'motifs'
    # ds = RNAPDBDataset("data/user_inputs/", name='test-pkl', mode='coarse-grain')
    # ds = RNAPDBDataset("data/rna3db-300/", name=dataset, mode='coarse-grain')
    ds = RNAPDBDataset("data/full_PDB/", name=dataset, mode='coarse-grain')
    # ds = RNAPDBDataset("data/RNA-PDB-clean/", name='test-pkl', mode='coarse-grain')
    ds_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    sampler = Sampler(timesteps=args.timesteps)
    print("Sampling...")
    sample(model, ds_loader, device, sampler, epoch, args, num_batches=None, exp_name=f"{exp_name}-{dataset}-seed={args.seed}")
    print(f"Results stored in path: samples/{exp_name}")

if __name__ == "__main__":
    main()