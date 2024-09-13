# RNAgrail: graph neural network and diffusion model for RNA 3D structure prediction

Welcome to the official repository of RNAgrail, a novel approach for RNA 3D structure prediction based on **graph neural networks (GNNs)** and **denoising diffusion models**. Unlike traditional methods that rely heavily on multiple sequence alignment (MSA) and structural templates, RNAgrail utilizes **local RNA descriptors**, making it particularly effective in scenarios where the data is limited and highly imbalanced.

## Table of contents
* [Overview](#overview)
* [Model architecture](#model-architecture)
* [Installation](#installation)
* [Inference](#inference)
* [Training](#training)
* [Evaluation](#evaluation)
* [Citation](#citation)
* [License](#license)



## Overview
RNAgrail leverages a graph-based approach combined with a denoising diffusion probabilistic model (DDPM) to predict the 3D structures of RNA at a fine-grained level. The core innovation of our model is the use of local RNA descriptors, which allows us to predict substructures of RNA rather than the full molecule at once. This tamplate-free approach was trained on rRNA and tRNA structures and generalizes to unseen RNA families. 

## Model architecture
RNAgrail's architecture is based on GNNs combined with a diffusion model. Below is a visualization of the model architecture:
<img title="Model architecture" alt="Architecture of RNAgrail." src="model-overview.png">

## Installation
Clone this repository and install the required dependencies using the following commands:
```
git clone git@github.com:mjustynaPhD/RNAgrail.git
cd RNAgrail
conda env create --name gnn -f environment.yml
```

Ensure you have **Python 3.10+** and **PyTorch 2.3.0+** installed, along with **PyTorch Geometric** for handling the graph components.

#### Install RiNALMo

RiNALMo is a key component of RNAgrail. To install that follow the instruction:
```
git clone https://github.com/lbcb-sci/RiNALMo
cd RiNALMo
pip install .
pip install flash-attn==2.3.2
```


#### Install SimRNA

For inference or building the output PDB file from predicted coordinates we use **SimRNA**. You can download SimRNA from [here](https://genesilico.pl/software/simrna/version_3.20/). In our experiments we used version `SimRNA_64bitIntel_Linux_staticLibs_withoutOpenMP` as we do not need to run anything in parallel. Then extract the archive in location you want.

## Inference

#### Configuration for inference
To run inference you should set SimRNA installation path in file `prepare_user_input.py`:
```
SIM_RNA = "/home/<your_path>/software/SimRNA_64bitIntel_Linux_staticLibs_withoutOpenMP"
```

#### Create your input.
Your input file should be in a ***.dotseq** format. The format looks as the example below:
```
>tsh_helix
CGCGGAACG CGGGACGCG
((((...(( ))...))))
```
First line is the name of the sequence, starting with `>`. Next line, is nucleotide sequence. Each strand is separated with white-space. The last line is a 2D structure in dot-bracket format. Similarly to sequence, each strand is separated by white-space. The following characters are allowed to mark interactions (including pseudoknots):
```
(), [], {}, <>, Aa, Bb, Cc, Dd
```

#### Prepare the file.
Run command:
```
python prepare_user_input.py --input-dir=user_inputs
```

The pickle files will be created in the directory `data/user_inputs`.

#### Run model
Then, to inference all structures from `data/user_inputs` directory run:
```
python sample_rna_pdb.py --seed=0 --batch_size=32 --dim=256 --n_layer=6 --timesteps=5000 --knns=20
```

## Training
If you wish to run training follow the instruction below.

#### Data preparation
First you need to preprocess your raw PDB/CIF files. For this purpose you need to run the following script:
```
python preprocess_rna_pdb.py --pdb_dir=<pdb_directory> --save_dir=data/<dataset name> --save_name=<train-pkl / test-pkl / val-pkl>
```

This script loads structures from PDB files and stores these files in pickle format. These files contain a molecule graph with all nodes and edges properties.

#### Run training
Once your data are preprocessed you are ready to run training. You can run it on a single GPU using the following command:
```
python main_rna_pdb_single.py --dataset <your dataset> --epoch=800 --batch_size=32 --dim=256 --n_layer=6 --lr=1e-3 --timesteps=5000 --mode=coarse-grain --knn=20 --wandb --lr-step=30
```
Or you can run it in a distributed system with the following bash script:
```
./run_distributed.sh
```
Note that the distributed script was created for a SLURM-based environement, so you will need to adapt it to your needs.

## Evaluation
In order to run evaluation you will need to run the following script:
```
python evaluate_predictions.py --preds-path=/home/<your path to RNAgrail>samples/<experiment name>/<epoch>/, \
                "--templates-path=<path to pdb templates>", \
                "--targets-path=<path to pdb targets>", \
                --sim_rna=/home/<path to SimRNA>/SimRNA_64bitIntel_Linux_staticLibs_withoutOpenMP",
```

The key arguments:
* `preds-path` - path to models predictions. This is generated during training or when the user run an inference.
* `templates-path` - path to PDB file (e.g. generated by SimRNA). The predicted coordinates will be substituted in this model.
* `targets-path` - path to ground-truth PDBs. The structures predicted structures and targets will be aligned and the metrics will be computed (RMSD, eRMSD and INF).
* `sim_rna` - path to SimRNA package to generate full-atom structure from coordinates.

## Citation
## License