# GraphaRNA Kinematic Refinement - Training Scripts

Comprehensive shell scripts for running the complete RNA 3D structure prediction pipeline using **Recursive Kinematic Refinement** with **Curriculum Noise Scheduling** and **Reinforcement Learning**.

---

## 📋 Overview

This pipeline implements a paradigm shift from Cartesian point-cloud diffusion to SE(3) frame-based kinematic refinement:

- **🔄 Kinematic Frames**: Each RNA residue = rigid body (R ∈ SO(3), t ∈ ℝ³) + 5 local CG atoms
- **🧠 VN-EGNN**: Vector-neuron enhanced E(n)-equivariant GNN predicting SE(3) updates
- **📚 Curriculum Learning**: Adaptive noise schedule σ starting at 0.05 rad (~3°), increasing when BP recovery > 80%
- **🎯 RL Fine-tuning**: PPO with physics-based reward (base pairs - clashes - torsion stress)
- **🔁 Recursive Refinement**: Self-correction loop with LDDT-based masking and lever damping

---

## 🚀 Quick Start

### Test the installation (2 minutes)
```bash
bash run_quick_test.sh full-3d
```

### Full pipeline from preprocessing to trained model
```bash
bash run_full_pipeline.sh full-3d
```

### Custom configuration
```bash
GPU=1 WANDB=1 SUP_EPOCHS=150 RL_EPOCHS=100 bash run_full_pipeline.sh P-only
```

---

## 📜 Available Scripts

### 1. **`run_preprocess.sh`** - Dataset Preprocessing

Augments existing PyG pickled datasets with kinematic frame data.

```bash
bash run_preprocess.sh [DATASET]
```

**Arguments:**
- `DATASET`: Dataset name (default: `full-3d`)

**Environment variables:**
- `GPU`: GPU device ID (default: 0)

**Example:**
```bash
bash run_preprocess.sh full-3d
GPU=1 bash run_preprocess.sh P-only
```

**What it does:**
- Computes rigid frames (R, t) from CG coordinates via Gram-Schmidt
- Extracts local atom coordinates in each frame's reference system
- Augments `{train,val,test}-pkl/` directories with frame data

---

### 2. **`run_supervised.sh`** - Supervised Pre-training

Phase 1: Train KinematicGNN + LDDTSuggester with curriculum noise.

```bash
bash run_supervised.sh [DATASET] [EPOCHS] [LR] [BATCH_SIZE]
```

**Arguments:**
- `DATASET`: Dataset name (default: `full-3d`)
- `EPOCHS`: Training epochs (default: 100)
- `LR`: Learning rate (default: `1e-3`)
- `BATCH_SIZE`: Batch size (default: 4)

**Environment variables:**
- `GPU`: GPU device (default: 0)
- `SEED`: Random seed (default: 42)
- `EXP_NAME`: Experiment name (default: `kinematic_supervised`)
- `WANDB`: Enable WandB logging (`1` or `true`)
- Model architecture: `NODE_DIM`, `HIDDEN_DIM`, `N_LAYERS`, etc.

**Examples:**
```bash
# Default settings
bash run_supervised.sh full-3d 100 1e-3 4

# Custom architecture with WandB
GPU=1 WANDB=1 NODE_DIM=256 N_LAYERS=6 bash run_supervised.sh full-3d 150 5e-4 8

# Quick test
bash run_supervised.sh P-only 20 1e-3 2
```

**Curriculum noise schedule:**
- σ starts at **0.05 rad (~3°)** on SO(3)
- When BP recovery ≥ **80%**, σ multiplies by **1.3×**
- Capped at **σ_max = 1.5 rad (~86°)**
- Forces GNN to learn **local repair** before **global folding**

**Output:**
- Checkpoints: `save/EXP_NAME/supervised_epoch_{10,20,...,EPOCHS}.pt`
- Training history with loss curves and curriculum σ tracking

---

### 3. **`run_rl.sh`** - RL Fine-tuning

Phase 2: Fine-tune with PPO using physics-based reward.

```bash
bash run_rl.sh CHECKPOINT [DATASET] [EPOCHS] [LR] [BATCH_SIZE]
```

**Arguments:**
- `CHECKPOINT`: Path to supervised checkpoint (required)
- `DATASET`: Dataset name (default: `full-3d`)
- `EPOCHS`: RL epochs (default: 50)
- `LR`: Learning rate (default: `3e-4`)
- `BATCH_SIZE`: Batch size (default: 4)

**Environment variables:** Same as `run_supervised.sh`

**Examples:**
```bash
# Standard RL fine-tuning
bash run_rl.sh save/kinematic_supervised/supervised_epoch_100.pt

# Extended RL with custom settings
GPU=1 bash run_rl.sh save/exp/supervised_epoch_150.pt full-3d 100 1e-4 8

# Resume from previous RL checkpoint
bash run_rl.sh save/kinematic_rl/rl_epoch_50.pt full-3d 50
```

**Reward function:**
```
R = w_bp · BP_recovery - w_clash · Clashes - w_torsion · TorsionStress
```
- **BP recovery**: Gaussian-scored N1/N9 distances (< 12 Å)
- **Clashes**: Pairwise P-atom distance penalty
- **Torsion**: P-P-P angle deviation from ideal RNA backbone

**Output:**
- Checkpoints: `save/EXP_NAME/rl_epoch_{10,20,...,EPOCHS}.pt`
- Curriculum continues: σ increases adaptively until max difficulty

---

### 4. **`run_hybrid.sh`** - Full Hybrid Training

Runs both supervised and RL phases sequentially in one pipeline.

```bash
bash run_hybrid.sh [DATASET] [SUP_EPOCHS] [RL_EPOCHS] [BATCH_SIZE]
```

**Arguments:**
- `DATASET`: Dataset name (default: `full-3d`)
- `SUP_EPOCHS`: Supervised epochs (default: 100)
- `RL_EPOCHS`: RL epochs (default: 50)
- `BATCH_SIZE`: Batch size (default: 4)

**Environment variables:**
- `SUP_LR`: Supervised learning rate (default: `1e-3`)
- `RL_LR`: RL learning rate (default: `3e-4`)
- All other variables from `run_supervised.sh`

**Examples:**
```bash
# Standard hybrid training
bash run_hybrid.sh full-3d 100 50 4

# Extended training with WandB
GPU=1 WANDB=1 bash run_hybrid.sh P-only 150 100 8

# Quick test
SUP_EPOCHS=20 RL_EPOCHS=10 bash run_hybrid.sh full-3d
```

**Output:**
- Two checkpoint directories:
  - `save/{EXP_NAME}_supervised/`
  - `save/{EXP_NAME}_rl/`

---

### 5. **`run_sample.sh`** - Structure Sampling

Generate RNA 3D structures from a trained model.

```bash
bash run_sample.sh CHECKPOINT INPUT_DOTSEQ
```

**Arguments:**
- `CHECKPOINT`: Path to trained model checkpoint (required)
- `INPUT_DOTSEQ`: Input file with sequences and secondary structures (required)

**Environment variables:**
- `GPU`: GPU device (default: 0)
- `EXP_NAME`: Output directory name (default: `kinematic_sample`)
- Model architecture variables (must match checkpoint)

**Input format** (`.dotseq`):
```
>tRNA_Phe
GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA
(((((((..((((.........)))).(((((.......))))).....(((((.......))))))))))))
```

**Examples:**
```bash
# Sample from RL checkpoint
bash run_sample.sh save/kinematic_rl/rl_epoch_50.pt input.dotseq

# Sample from supervised checkpoint
bash run_sample.sh save/kinematic_supervised/supervised_epoch_100.pt data/test.dotseq

# Multiple GPU sampling
GPU=1 bash run_sample.sh save/exp/rl_epoch_100.pt sequences.dotseq
```

**Output:**
- `samples/EXP_NAME/{name}.pt` - Saved coordinates + pLDDT
- Can be converted to PDB with `frames_to_pdb` utility

---

### 6. **`run_full_pipeline.sh`** - Complete Pipeline

Master script: Preprocessing → Supervised → RL

```bash
bash run_full_pipeline.sh [DATASET]
```

**Arguments:**
- `DATASET`: Dataset name (default: `full-3d`)

**Environment variables:**
- All variables from previous scripts
- `SKIP_PREPROCESS`: Skip preprocessing if already done (default: 0)

**Examples:**
```bash
# Full pipeline with defaults
bash run_full_pipeline.sh full-3d

# Custom configuration
GPU=1 WANDB=1 SUP_EPOCHS=150 RL_EPOCHS=100 BATCH_SIZE=8 bash run_full_pipeline.sh P-only

# Skip preprocessing (if already done)
SKIP_PREPROCESS=1 bash run_full_pipeline.sh full-3d

# Smaller model for testing
NODE_DIM=64 HIDDEN_DIM=64 N_LAYERS=2 bash run_full_pipeline.sh full-3d
```

**Pipeline steps:**
1. **Preprocess** - Augment dataset with frames
2. **Supervised** - 100 epochs with curriculum
3. **RL** - 50 epochs with PPO

**Output:**
- Log file: `pipeline_{DATASET}_{TIMESTAMP}.log`
- Final checkpoint: `save/{EXP_NAME}_rl/rl_epoch_{RL_EPOCHS}.pt`

---

### 7. **`run_quick_test.sh`** - Fast Validation

Minimal training run for testing installation and debugging.

```bash
bash run_quick_test.sh [DATASET]
```

**Settings:**
- 10 supervised epochs
- 5 RL epochs
- Batch size 2
- Small model (64-dim, 2 layers)

**Examples:**
```bash
bash run_quick_test.sh full-3d
GPU=1 bash run_quick_test.sh P-only
```

**Use cases:**
- Testing installation
- Debugging pipeline issues
- Quick architecture experiments
- CI/CD validation

---

## ⚙️ Configuration Reference

### Model Architecture

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_DIM` | 128 | Node feature dimension |
| `EDGE_DIM` | 64 | Edge feature dimension |
| `HIDDEN_DIM` | 128 | Hidden layer dimension |
| `N_LAYERS` | 4 | Number of VN-EGNN layers |
| `N_VECTOR_FEATURES` | 8 | Vector neuron features |
| `MAX_REFINEMENT_STEPS` | 4 | Recursive refinement iterations |
| `LEVER_DAMPING` | 0.95 | Damping factor for forward kinematics |

### Training Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `SUP_EPOCHS` | 100 | Supervised training epochs |
| `RL_EPOCHS` | 50 | RL fine-tuning epochs |
| `SUP_LR` | 1e-3 | Supervised learning rate |
| `RL_LR` | 3e-4 | RL learning rate |
| `BATCH_SIZE` | 4 | Training batch size |
| `SEED` | 42 | Random seed |

### Curriculum Scheduler (hardcoded in `curriculum_scheduler.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sigma_init` | 0.05 | Initial SO(3) noise (rad) |
| `sigma_max` | 1.5 | Maximum SO(3) noise (rad) |
| `sigma_growth_factor` | 1.3 | Multiplicative increase |
| `bp_threshold` | 0.80 | BP recovery threshold for promotion |
| `translation_ratio` | 3.0 | Translation noise scale |
| `warmup_steps` | 20 | Min steps before promotion |
| `reward_window` | 50 | Rolling average window |

---

## 📊 Monitoring Training

### With WandB

```bash
WANDB=1 bash run_hybrid.sh full-3d
```

Logs:
- Training loss curves
- BP recovery rate
- Current curriculum σ
- Difficulty progress (0-100%)
- RL reward components
- pLDDT distributions

### Without WandB

All metrics are logged to console and saved in checkpoint history:
```python
checkpoint = torch.load('save/exp/supervised_epoch_100.pt')
print(checkpoint.keys())  # ['epoch', 'phase', 'gnn_state_dict', 'lddt_state_dict', 'config', 'curriculum_scheduler']
```

---

## 🔧 Hardware Requirements

### Minimal (Quick Test)
- **GPU**: 6GB VRAM (GTX 1060 / RTX 2060)
- **RAM**: 16GB
- **Storage**: 10GB
- **Time**: ~5 minutes

### Recommended (Full Training)
- **GPU**: 16GB VRAM (V100 / RTX 3090 / A5000)
- **RAM**: 32GB
- **Storage**: 50GB
- **Time**: ~6-12 hours (100 sup + 50 RL epochs)

### Large-Scale
- **GPU**: 40GB VRAM (A100)
- **RAM**: 64GB+
- **Batch size**: 16-32
- **Time**: ~3-6 hours with larger batches

---

## 📁 Directory Structure

```
GraphaRNA/
├── data/
│   ├── full-3d/
│   │   ├── train-pkl/      # Training set (pickled PyG graphs)
│   │   ├── val-pkl/        # Validation set
│   │   └── test-pkl/       # Test set
│   └── P-only/             # Phosphate-only dataset
├── save/                    # Checkpoints
│   └── {EXP_NAME}/
│       ├── supervised_epoch_*.pt
│       └── rl_epoch_*.pt
├── samples/                 # Generated structures
│   └── {EXP_NAME}/
│       └── *.pt
├── src/grapharna/kinematic/ # Kinematic refinement modules
│   ├── frames.py            # SE(3) frame representation
│   ├── kinematic_gnn.py     # VN-EGNN model
│   ├── curriculum_scheduler.py  # Adaptive noise schedule
│   ├── rl_agent.py          # PPO agent
│   ├── recursive_refiner.py # Self-correction loop
│   └── train_hybrid.py      # Training loop
├── run_preprocess.sh        # Preprocessing script
├── run_supervised.sh        # Supervised training
├── run_rl.sh                # RL fine-tuning
├── run_hybrid.sh            # Supervised + RL
├── run_sample.sh            # Structure sampling
├── run_full_pipeline.sh     # Complete pipeline
├── run_quick_test.sh        # Fast validation
└── TRAINING_SCRIPTS.md      # This file
```

---

## 🐛 Troubleshooting

### CUDA out of memory
```bash
BATCH_SIZE=2 bash run_hybrid.sh full-3d
```

### Import errors
```bash
pip install -e .                    # Install GraphaRNA package
pip install -r requirements.txt     # Install dependencies
```

### Preprocessing takes too long
```bash
# Only preprocess train/val splits
ls data/full-3d/test-pkl/ && rm -rf data/full-3d/test-pkl/
bash run_preprocess.sh full-3d
```

### Resume from checkpoint
```bash
# For supervised training
bash run_supervised.sh full-3d 200  # continues from epoch 100 if checkpoint exists

# For RL
bash run_rl.sh save/kinematic_supervised/supervised_epoch_100.pt full-3d 100
```

### Skip preprocessing
```bash
SKIP_PREPROCESS=1 bash run_full_pipeline.sh full-3d
```

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{GraphaRNA2025,
  title={GraphaRNA: Recursive Kinematic Refinement for RNA 3D Structure Prediction},
  author={...},
  journal={Bioinformatics},
  year={2025}
}
```

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Last updated:** February 2026
