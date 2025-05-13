# NoProp: Training Neural Networks Without Back-Propagation

[![arXiv](https://img.shields.io/badge/arXiv-2503.24322-b31b1b.svg)](https://arxiv.org/abs/2503.24322)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the NoProp learning method from the paper ["NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION"](https://arxiv.org/abs/2503.24322) (Qinyu Li, Yee Whye Teh, Razvan Pascanu). This project explores training neural network layers independently by learning to denoise noisy targets, inspired by diffusion and flow matching methods.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Core Concept](#core-concept)
- [Implementation Details](#implementation-details)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Overview

NoProp offers an alternative to traditional back-propagation by training each layer independently. Key features:
- No forward/backward propagation during training
- Denoising-based learning approach
- Fixed intermediate representations
- Iterative denoising for inference

## Project Structure

```
NoProp/
├── checkpoints_run/        # Model checkpoints and training logs
├── data/                   # Dataset storage
├── models/                 # Model architecture definitions
│   ├── NoPropBlock.py     # NoPropBlock module (û_θ_t)
│   └── NoProp_DT.py       # NoPropNetDT model
├── data.py                # Data loading and preprocessing
├── main.py                # Training and evaluation script
├── requirements.txt       # Project dependencies
├── train.py              # Training utilities
└── visualize.py          # Visualization tools
```

## Core Concept

### Traditional vs NoProp Approach

| Aspect | Traditional | NoProp |
|--------|-------------|---------|
| Training Flow | Sequential forward/backward | Independent block training |
| Learning Process | Error propagation | Denoising noisy targets |
| Layer Dependencies | Strong | None |
| Parallelization | Limited | High |

### Key Components

1. **Denoising Blocks**: Each block learns to predict clean target embeddings
2. **Noise Schedule**: Cosine schedule for controlled noise addition
3. **Embedding Strategies**:
   - One-hot (fixed)
   - Learned (trainable)
   - Prototype (data-initialized)

## Implementation Details

### Model Architecture
- **NoPropNetDT**: Main network with T denoising blocks
- **NoPropBlock**: Core denoising unit with CNN and MLP pathways

### Training Process
- Independent block training
- Cosine noise schedule
- Learning rate warmup
- Early stopping
- Checkpoint management

### Visualization Tools
- Training history plots
- Embedding visualizations
- Confusion matrices
- Prediction analysis

## 🚀 Getting Started

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
python main.py \
    --embedding-type learned \
    --embedding-dim 64 \
    --num-blocks 10 \
    --batch-size 128 \
    --epochs 150 \
    --lr 0.001 \
    --weight-decay 0.0001
```

## Results

Training results are stored in `checkpoints_run/` with:
- Model checkpoints
- Training history logs
- Visualization plots

Use `visualize.py` to analyze results:
```bash
python visualize.py --history path/to/training_history.json
```

## Citation

```bibtex
@misc{li2025noprop,
      title={NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION},
      author={Qinyu Li and Yee Whye Teh and Razvan Pascanu},
      year={2025},
      eprint={2503.24322},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
