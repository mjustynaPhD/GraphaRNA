#!/usr/bin/env python3
"""
Entry point for Recursive Kinematic Refinement training.

Usage:
    # Supervised pre-training
    python -m grapharna.kinematic.main --phase supervised --dataset full-3d

    # RL fine-tuning (after supervised)
    python -m grapharna.kinematic.main --phase rl --checkpoint save/exp/supervised_epoch_100.pt

    # Full hybrid (supervised → RL)
    python -m grapharna.kinematic.main --phase hybrid --dataset full-3d

    # Inference / sampling
    python -m grapharna.kinematic.main --phase sample --checkpoint save/exp/rl_epoch_50.pt --input input.dotseq
"""

import argparse
import logging
import os
import os.path as osp
import random

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from grapharna.datasets import RNAPDBDataset
from grapharna.kinematic.kinematic_gnn import KinematicConfig
from grapharna.kinematic.train_hybrid import HybridTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphaRNA — Recursive Kinematic Refinement"
    )

    # Phase
    parser.add_argument(
        '--phase', type=str, default='hybrid',
        choices=['supervised', 'rl', 'hybrid', 'sample', 'preprocess'],
        help='Training phase: supervised, rl, hybrid, sample, or preprocess'
    )

    # Data
    parser.add_argument('--dataset', type=str, default='full-3d')
    parser.add_argument('--mode', type=str, default='coarse-grain')
    parser.add_argument('--input', type=str, default=None,
                        help='Input .dotseq file for sampling')

    # Model architecture
    parser.add_argument('--node-dim', type=int, default=128)
    parser.add_argument('--edge-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-vector-features', type=int, default=8)
    parser.add_argument('--spatial-knn', type=int, default=16)
    parser.add_argument('--spatial-cutoff', type=float, default=20.0)
    parser.add_argument('--max-refinement-steps', type=int, default=4)
    parser.add_argument('--lever-damping', type=float, default=0.95)
    parser.add_argument('--transformer-blocks', type=int, default=4)

    # Training
    parser.add_argument('--supervised-epochs', type=int, default=100)
    parser.add_argument('--rl-epochs', type=int, default=50)
    parser.add_argument('--supervised-lr', type=float, default=1e-3)
    parser.add_argument('--rl-lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='./save')
    parser.add_argument('--exp-name', type=str, default=None)

    # Logging
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    logger.info(f"Device: {device}")

    # Build config
    config = KinematicConfig(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_vector_features=args.n_vector_features,
        spatial_knn=args.spatial_knn,
        spatial_cutoff=args.spatial_cutoff,
        max_refinement_steps=args.max_refinement_steps,
        lever_damping=args.lever_damping,
        transformer_blocks=args.transformer_blocks,
    )

    exp_name = args.exp_name or f"kinematic_{args.phase}"

    # ---- Preprocessing phase ----
    if args.phase == 'preprocess':
        from grapharna.kinematic.preprocess_frames import batch_augment_dataset
        data_path = osp.join('.', 'data', args.dataset)
        for split in ['train-pkl', 'val-pkl', 'test-pkl']:
            split_path = osp.join(data_path, split)
            if osp.exists(split_path):
                logger.info(f"Augmenting {split_path} with frame data...")
                batch_augment_dataset(split_path)
        return

    # ---- Load data ----
    data_path = osp.join('.', 'data', args.dataset)
    logger.info(f"Loading dataset from {data_path}")

    train_dataset = RNAPDBDataset(data_path, name='train-pkl', mode=args.mode)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_loader = None
    val_path = osp.join(data_path, 'val-pkl')
    if osp.exists(val_path):
        val_dataset = RNAPDBDataset(data_path, name='val-pkl', mode=args.mode)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    logger.info(f"Train: {len(train_dataset)} structures")
    if val_loader:
        logger.info(f"Val:   {len(val_dataset)} structures")

    # ---- WandB ----
    if args.wandb:
        import wandb
        wandb.login()
        run = wandb.init(project='GraphaRNA-Kinematic', config=vars(args))
        exp_name = run.name

    # ---- Build trainer ----
    trainer = HybridTrainer(
        config=config,
        supervised_epochs=args.supervised_epochs,
        rl_epochs=args.rl_epochs,
        supervised_lr=args.supervised_lr,
        rl_lr=args.rl_lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        device=device,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # ---- Run phase ----
    if args.phase == 'supervised':
        history = trainer.train_supervised(
            train_loader, val_loader, exp_name
        )
    elif args.phase == 'rl':
        history = trainer.train_rl(train_loader, exp_name)
    elif args.phase == 'hybrid':
        trainer.train_hybrid(
            train_loader, val_loader, exp_name
        )
    elif args.phase == 'sample':
        _run_sampling(trainer, args, config, device)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")

    logger.info("Done!")


def _run_sampling(trainer, args, config, device):
    """Run inference / structure generation."""
    from grapharna.utils import read_dotseq_file
    from grapharna.constants import RESIDUES

    if args.input:
        # Parse dotseq file
        names, sequences_list, dot_brackets = read_dotseq_file(args.input)
    else:
        raise ValueError("--input is required for sampling phase")

    trainer.refiner.eval()

    for name, seqs, dot in zip(names, sequences_list, dot_brackets):
        logger.info(f"Generating structure for {name}")
        full_seq = "".join(seqs)
        N = len(full_seq)

        residue_types = torch.tensor(
            [RESIDUES.get(nt, 0) for nt in full_seq],
            dtype=torch.long,
            device=device,
        )
        batch = torch.zeros(N, dtype=torch.long, device=device)
        chain_edges = torch.stack([
            torch.arange(N - 1, device=device),
            torch.arange(1, N, device=device),
        ])

        # Parse base pairs from dot-bracket
        from grapharna.preprocess_rna_pdb import dot_to_bpseq
        bp_list = dot_to_bpseq(dot)
        if bp_list:
            bp_src = torch.tensor([bp[0] for bp in bp_list], device=device)
            bp_dst = torch.tensor([bp[1] for bp in bp_list], device=device)
            bp_edges = torch.stack([bp_src, bp_dst])
        else:
            bp_edges = torch.zeros(2, 0, dtype=torch.long, device=device)

        # Generate
        result = trainer.refiner.sample(
            sequences=seqs if isinstance(seqs, list) else [full_seq],
            residue_types=residue_types,
            batch=batch,
            covalent_edges=chain_edges,
            bp_edges=bp_edges,
            chain_edges=chain_edges,
            device=device,
        )

        # Convert to PDB
        coords = result['coords']  # (N, 5, 3)
        plddt = result['plddt']
        logger.info(
            f"  {name}: mean pLDDT = {plddt.mean():.3f}, "
            f"iterations = {result['n_iterations']}"
        )

        # Save output
        out_dir = f"./samples/{args.exp_name or 'kinematic_sample'}"
        os.makedirs(out_dir, exist_ok=True)
        # Flatten to (N*5, 15) format for SampleToPDB compatibility
        # This would need adaptation for the new frame representation
        # For now, save raw coordinates
        torch.save({
            'coords': coords.cpu(),
            'plddt': plddt.cpu(),
            'sequence': full_seq,
            'name': name,
        }, os.path.join(out_dir, f"{name}.pt"))

        logger.info(f"  Saved to {out_dir}/{name}.pt")


if __name__ == "__main__":
    main()
