"""
Preprocessing extension — stores local reference frames for the 5-atom CG model.

Extends the existing preprocessing pipeline to compute and store
per-residue SE(3) frames alongside the graph representation.
"""

import torch
import numpy as np
import pickle
import os
from typing import Dict, Optional

from grapharna.kinematic.frames import (
    compute_local_frames_from_coords,
    IDEAL_LOCAL_COORDS,
)


def augment_pickle_with_frames(
    pkl_path: str,
    output_path: Optional[str] = None,
) -> Dict:
    """Load a preprocessed .pkl file and compute reference frames.

    Adds the following keys to the pickle:
        'frame_R':           (N_res, 3, 3)  rotation matrices
        'frame_t':           (N_res, 3)     frame origins (P positions)
        'frame_local_coords': (N_res, 5, 3) atoms in local frame
        'residue_indices':   (N_res,)       residue type indices

    Args:
        pkl_path:    Path to existing .pkl file.
        output_path: Where to save augmented pickle (default: overwrite).
    Returns:
        Augmented data dict.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    coords = np.array(data['pos'], dtype=np.float32)
    residues = data.get('residues', None)

    if residues is None:
        raise ValueError(f"No residue info in {pkl_path}")

    # Get number of residues (5 atoms per residue in CG model)
    N_atoms = len(coords)
    N_res = N_atoms // 5

    if N_atoms % 5 != 0:
        raise ValueError(
            f"Expected 5 atoms per residue, got {N_atoms} atoms "
            f"({N_atoms / 5:.1f} residues)"
        )

    # Reshape to (N_res, 5, 3)
    coords_reshaped = torch.tensor(coords).reshape(N_res, 5, 3)
    residue_types = torch.tensor(residues[::5]).long()  # One per residue

    # Compute frames
    R, t, local_coords = compute_local_frames_from_coords(
        coords_reshaped, residue_types
    )

    # Store in data dict
    data['frame_R'] = R.numpy()
    data['frame_t'] = t.numpy()
    data['frame_local_coords'] = local_coords.numpy()
    data['residue_indices'] = residue_types.numpy()

    # Save
    out_path = output_path or pkl_path
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)

    return data


def batch_augment_dataset(
    dataset_dir: str,
    output_dir: Optional[str] = None,
    file_extension: str = '.pkl',
):
    """Augment all .pkl files in a dataset directory with frame information.

    Args:
        dataset_dir: Directory containing .pkl files.
        output_dir:  Output directory (default: in-place).
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(dataset_dir) if f.endswith(file_extension)]
    n_success = 0
    n_failed = 0

    for fname in files:
        in_path = os.path.join(dataset_dir, fname)
        out_path = os.path.join(output_dir, fname) if output_dir else in_path

        try:
            augment_pickle_with_frames(in_path, out_path)
            n_success += 1
        except Exception as e:
            print(f"Failed to process {fname}: {e}")
            n_failed += 1

    print(f"Augmented {n_success} files, {n_failed} failures")


def compute_ideal_local_coords_from_dataset(
    dataset_dir: str,
    file_extension: str = '.pkl',
) -> Dict[str, torch.Tensor]:
    """Compute mean local coordinates per residue type from a dataset.

    This refines the IDEAL_LOCAL_COORDS constants using actual
    structural data.

    Args:
        dataset_dir: Directory containing augmented .pkl files.
    Returns:
        Dict mapping residue name → (5, 3) mean local coordinates.
    """
    from grapharna.constants import REV_RESIDUES

    accumulators = {name: [] for name in ['A', 'G', 'U', 'C']}

    files = [f for f in os.listdir(dataset_dir) if f.endswith(file_extension)]

    for fname in files:
        path = os.path.join(dataset_dir, fname)
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            if 'frame_local_coords' not in data:
                continue

            local_coords = torch.tensor(data['frame_local_coords'])
            residue_indices = data.get('residue_indices', None)

            if residue_indices is None:
                continue

            for i, res_idx in enumerate(residue_indices):
                res_name = REV_RESIDUES.get(int(res_idx), None)
                if res_name in accumulators:
                    accumulators[res_name].append(local_coords[i])

        except Exception:
            continue

    ideal_coords = {}
    for name, coords_list in accumulators.items():
        if len(coords_list) > 0:
            stacked = torch.stack(coords_list)
            ideal_coords[name] = stacked.mean(dim=0)
            std = stacked.std(dim=0).mean().item()
            print(f"{name}: {len(coords_list)} samples, mean std = {std:.3f} Å")
        else:
            ideal_coords[name] = IDEAL_LOCAL_COORDS.get(name, torch.zeros(5, 3))

    return ideal_coords
