# LeakGFN

**Leaky GFlowNet for Molecular Optimization**

This repository contains the official implementation of LeakGFN, a GFlowNet-based framework for molecular optimization with leaky exploration mechanism.

## Overview

LeakGFN extends the standard GFlowNet framework with a leaky exploration mechanism to improve diversity and sample efficiency in molecular optimization tasks. The framework supports multiple training objectives including:

- **JNK3**: c-Jun N-terminal kinase 3 inhibition
- **GSK3β**: Glycogen synthase kinase 3 beta inhibition  
- **DRD2**: Dopamine receptor D2 binding affinity
- **QED**: Quantitative Estimate of Drug-likeness
- **SA**: Synthetic Accessibility

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- torch-geometric
- torch-scatter
- RDKit
- NumPy
- Pandas
- PyYAML
- wandb (for experiment tracking)

### Installation

```bash
# Clone the repository
# Download the repository from:
# https://anonymous.4open.science/r/LeakGFN-464A
# Then extract and navigate to the directory

cd LeakGFN

# Create conda environment
conda env create --file conda_install.yaml

# Activate the environment
conda activate LeakGFN

# Install pip dependencies
pip install -r pip_requirement.txt
```

## Usage

### Basic Training

Train LeakGFN on a specific molecular optimization task:

```bash
python train.py --config_file ./configs/JNK3.yaml --seed 1 --log_dir ./checkpoints/LeakGFN/seed_1
```

### Available Configurations

| Config File | Target | Description |
|-------------|--------|-------------|
| `JNK3.yaml` | JNK3 | c-Jun N-terminal kinase 3 inhibitor |
| `GSK3B.yaml` | GSK3β | Glycogen synthase kinase 3 beta inhibitor |
| `DRD2.yaml` | DRD2 | Dopamine receptor D2 binding |
| `QED.yaml` | QED | Drug-likeness optimization |
| `SA.yaml` | SA | Synthetic accessibility optimization |

### Training with Different Criteria

LeakGFN supports multiple GFlowNet training criteria:

```bash
bash train.sh

```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config_file` | - | Path to YAML configuration file |
| `--seed` | 42 | Random seed for reproducibility |
| `--log_dir` | `./checkpoints` | Directory to save checkpoints and logs |
| `--device` | `cuda` | Device to use (`cuda` or `cpu`) |
| `--num_iterations` | 30000 | Number of training iterations |
| `--criterion` | `FM` | Training criterion (`FM`, `TB`, `SubTB`) |
| `--learning_rate` | 5e-4 | Learning rate |
| `--min_blocks` | 2 | Minimum number of building blocks |
| `--max_blocks` | 8 | Maximum number of building blocks |
| `--reward_exp` | 12 | Reward exponent for shaping |

## Project Structure

```
LeakGFN/
├── train.py                    # Main training script
├── configs/                    # Configuration files
│   ├── JNK3.yaml
│   ├── GSK3B.yaml
│   ├── DRD2.yaml
│   ├── QED.yaml
│   └── SA.yaml
├── gflownet/
│   ├── generator/              # GFlowNet model implementations
│   │   ├── leakgfn.py # Leaky GFlowNet (main)
│   │   └── gfn.py
│   ├── oracle/                 # Molecular scoring oracles
│   │   ├── oracle.py
│   │   ├── models.py
│   │   └── scorer/
│   ├── data/                   # Building blocks and test data
│   │   ├── blocks_105.json
│   │   └── test_mols_6062.pkl.gz
│   ├── utils/                  # Utility functions
│   │   ├── arguments.py
│   │   ├── metrics.py
│   │   ├── logging.py
│   │   └── utils.py
│   ├── mol_mdp_ext.py          # Molecular MDP definition
│   ├── model_block.py          # Block-based molecular model
│   └── model_atom.py           # Atom-based molecular model
├── checkpoints/                # Saved model checkpoints
└── results.ipynb               # Results analysis notebook
```

## Configuration

Configuration files (YAML) support the following key parameters:

```yaml
# General Settings
device: 'cuda'
seed: 42
save: True

# Objectives
objectives: 'jnk3'  # Single objective

# GFlowNet Settings
min_blocks: 2
max_blocks: 8
num_iterations: 30000
criterion: 'FM'
learning_rate: 0.0005

# Reward Shaping
reward_min: 0.01
reward_norm: 0.2
reward_exp: 12

# Architecture
repr_type: 'block_graph'
nemb: 256
num_conv_steps: 8
```

## Experiment Tracking

LeakGFN integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking:

```bash
# Enable wandb logging (default)
python train.py --config_file ./configs/JNK3.yaml

# Disable wandb (debug mode)
python train.py --config_file ./configs/JNK3.yaml --debug True
```

## Citation

If you find this code useful in your research, please cite:

```bibtex
@article{leakgfn2026,
  title={LeakGFN: Leaky GFlowNet for Molecular Optimization},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This implementation builds upon the GFlowNet framework. We thank the authors of the original GFlowNet papers for their foundational work.
