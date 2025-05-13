# NoProp: Training Neural Networks Without Back-Propagation

[![arXiv](https://img.shields.io/badge/arXiv-2503.24322-b31b1b.svg)](https://arxiv.org/abs/2503.24322)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the NoProp learning method from the paper ["NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION"](https://arxiv.org/abs/2503.24322) (Qinyu Li, Yee Whye Teh, Razvan Pascanu). This project explores training neural network layers independently by learning to denoise noisy targets, inspired by diffusion and flow matching methods.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Core Concept](#core-concept)
- [Getting Started](#getting-started)
- [Usage](#usage)
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

## Core concept of NoProp

**NoProp** introduces an alternative training framework that obviates the need for end-to-end forward and backward signal propagation during the parameter update phase for individual network components.

1.  **Decentralized Component Training**:
    * A fundamental departure from back-propagation is that NoProp enables the constituent modules (termed "blocks" or layers) of the neural network to be trained **independently**.
    * This negates the requirement for global error signal propagation across the entire network depth for the training of each distinct block.

2.  **Learning via Denoising**:
    * The conceptual foundation of NoProp is derived from generative modeling techniques, particularly those related to denoising diffusion processes.
    * The primary learning objective for each block is to reconstruct a "clean" target representation (an embedding of the true class label) when provided with both the original input data and a **perturbed or "noisy" version of this target representation**.
    * This task effectively trains each block as a conditional denoising function.

3.  **Mechanism of Independent Block Training**:
    * Each block's parameters are optimized to enhance its denoising capability. The predicted clean representation is compared to the ground-truth clean representation, and the block's parameters are adjusted to minimize this discrepancy.
    * The noisy input representation supplied to a given block during its training is generated directly from the true clean target representation and a predefined noise schedule. This input is not the output of a preceding block in a sequential pass, thereby facilitating independent training.

4.  **Nature of Intermediate Representations**:
    * Unlike conventional deep learning where layers progressively learn and transform representations, NoProp operates with **predetermined intermediate target representations**.
    * During training, the input to each block (excluding the raw data input x) is a noisy version of the *final* target embedding, with the noise level dictated by a predefined schedule.
    * Consequently, blocks are not primarily learning to transform representations hierarchically but are specializing in denoising specific noise characteristics from a target, conditioned on the input data.

5.  **Inference Procedure**:
    * While training is characterized by independent block updates, the inference (prediction) phase is sequential.
    * The process commences with an initial state of pure noise.
    * This noisy state is then iteratively refined by passing it sequentially through the series of trained denoising blocks. Each block, conditioned on the input data, processes the output of the preceding block, aiming to reduce noise and move the state closer to a clean target representation.
    * Upon completion of T denoising steps, the resultant refined state is input to a final classification layer to produce the output prediction.



## Getting Started

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
