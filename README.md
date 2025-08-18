# BioEmb: Fine-Tuning Biochemical Embeddings with Self-Constrained Generative Transformers

This repository contains the official implementation for the paper: "BioEmb: Fine-Tuning Biochemical Embeddings with Self-Constrained Generative Transformers".

BioEmb is a novel framework for fine-tuning pre-trained models (like ESM for proteins and MolFormer for molecules) to create high-quality, domain-specific embeddings. Instead of using standard objectives like Masked Language Modeling (MLM), BioEmb employs a self-constrained generative task where the model learns to reconstruct an entity's own unique discrete sequence.

This method forces the model to learn a compressed, informative representation, resulting in embeddings that significantly outperform baselines on a variety of downstream tasks.

## Repository Structure

```
.
├── configs/
│   └── bbbp.yaml           # Example configuration file for the BBBP task
├── data/                   # (Optional) Directory for datasets
├── results/                # Directory for saving model checkpoints and outputs
├── bioemb/
│   ├── data_manager.py     # Pytorch Dataset and data loading utilities
│   ├── downstream_eval.py  # Downstream task evaluation (classifier training, metrics)
│   ├── models.py           # Core BioEmb model architecture
│   ├── pretrained.py       # Manages loading of pretrained encoders/tokenizers
│   ├── trie.py             # Prefix Trie implementation for constrained decoding
│   └── rvq.py              # Residual Vector Quantization implementation
├── train_bioemb.py         # Main script to train a BioEmb model
├── train_mlm_baseline.py   # Script to fine-tune an encoder using the MLM baseline
└── utils.py                # Utility functions (config parsing, logging callbacks)
```

## Setup

### 1\. Clone the repository:

```bash
git clone https://github.com/amitaysicherman/BioEmb.git
cd BioEmb
```

### 2\. Install dependencies:

We recommend using a conda environment.

```bash
conda create -n bioemb python=3.9
conda activate bioemb
pip install -r requirements.txt
```

*(Note: You will need to create a `requirements.txt` file)*

## Running Experiments

All experiments are driven by configuration files located in the `configs/` directory.

### 1\. Training a BioEmb Model

To train a BioEmb model, use the main training script and specify a configuration file.

```bash
python train_bioemb.py --config configs/bbbp.yaml
```

This will:

1.  Load the dataset specified in `bbbp.yaml`.
2.  Initialize the pre-trained encoder (e.g., MolFormer).
3.  Run the Residual Vector Quantization (RVQ) process to create discrete sequences.
4.  Build the prefix trie for constrained generation.
5.  Train the BioEmb model, saving checkpoints and logs to the `output_dir` specified in the config.
6.  Continuously evaluate performance on downstream tasks during training.

### 2\. Fine-tuning with the MLM Baseline

To run the MLM fine-tuning baseline for comparison, use the dedicated script:

```bash
python train_mlm_baseline.py --config configs/bbbp.yaml
```

This script will fine-tune the base encoder using a Masked Language Modeling objective on the same dataset.

