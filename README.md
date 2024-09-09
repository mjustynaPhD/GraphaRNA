# RNAgrail: graph neural network and diffusion model for RNA 3D structure prediction

Welcome to the official repository of RNAgrail, a novel approach for RNA 3D structure prediction based on **graph neural networks (GNNs)** and **denoising diffusion models**. Unlike traditional methods that rely heavily on multiple sequence alignment (MSA) and structural templates, RNAgrail utilizes **local RNA descriptors**, making it particularly effective in scenarios where the data is limited and highly imbalanced.

## Table of contents
* [Overview](#overview)
* [Model architecture](#model-architecture)
* [Installation](#installation)
* [Usage](#usage)
* [Data Preparation](#data-preparation)
* [Training](#training)
* [Inference](#inference)
* [Evaluation](#evaluation)
* [Citation](#citation)
* [License](#license)



## Overview
RNAgrail leverages a graph-based approach combined with a denoising diffusion probabilistic model (DDPM) to predict the 3D structures of RNA at a fine-grained level. The core innovation of our model is the use of local RNA descriptors, which allows us to predict substructures of RNA rather than the full molecule at once. This tamplate-free approach was trained on rRNA and tRNA structures and generalizes to unseen RNA families. 

## Model architecture
RNAgrail's architecture is based on GNNs combined with a diffusion model. Below is a visualization of the model architecture:
<img title="Model architecture" alt="Architecture of RNAgrail." src="model-overview.png">

## Installation
## Usage
## Data Preparation
## Training
## Inference
## Evaluation
## Citation
## License