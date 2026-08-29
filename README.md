<div align="center">

# HYPER: A Foundation Model for Inductive Link Prediction with Knowledge Hypergraphs

[![arXiv](https://img.shields.io/badge/arXiv-2506.12362-b31b1b.svg)](https://arxiv.org/abs/2506.12362)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG 2.4+](https://img.shields.io/badge/PyG-2.4%2B-3C2179?logo=python&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![Triton](https://img.shields.io/badge/Triton-2.1%2B-1a237e?logo=nvidia&logoColor=white)](https://triton-lang.org/)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)


</div>


A single pre-trained **HYPER** checkpoint achieves strong **zero-shot** link prediction performance across *arbitrary* knowledge hypergraphs and can be **fine-tuned** to further enhance accuracy.  
Unlike traditional knowledge graph (KG) models, **HYPER** natively supports **multi-relational, ordered hyperedges** of the form *(r(u₁, u₂, …))* without relying on qualifiers, and seamlessly accommodates both **transductive** and **inductive** evaluation settings.


---

## Table of Contents
1. [Why HYPER?](#why-hyper)
2. [Key Features](#key-features)
3. [Model Zoo](#model-zoo)
4. [Installation](#installation)
5. [Quick-Start](#quick-start)
6. [Running Experiments](#running-experiments)
7. [Datasets](#datasets)
8. [Pre-Training](#pre-training)
9. [Directory Tour](#directory-tour)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

---

## Why HYPER?
* **Hypergraph-native:** Works directly with n-ary, ordered hyper-edges—no awkward binary reification.
* **Foundation model:** A single model generalizes across 16+ hypergraphs
* **Inductive strong:** Supports node-inductive and node-relation-inductive settings without retraining embeddings.
* **Scalable GNN core:** Uses Triton-accelerated relational sparse-matrix multiplication (rspmm) for O(**V**) message passing, avoiding O(**E**) materialisation.
* **Flexible training recipes:** Pre-train on arbitrary mixtures of graphs, then fine-tune or directly infer.
* **PyTorch-Geometric 2.4+:** Pure Python, minimal dependencies, multi-GPU (DDP) ready.

---

## Key Features
| Feature | Description |
|---------|-------------|
| **Unified Relative Representations** | Entities and relations are **contextualised**—embeddings are computed on-the-fly from the hypergraph structure, not stored in lookup tables. |
| **Hypergraph Layer** | Custom `HypergraphLayer` implements efficient relational aggregation with sinusoidal position encodings and optional Triton-accelerated kernels. |
| **Data Pipeline** | `data_generation/` scripts convert raw triples to padded hyper-edge tensors; inductive splits follow GraIL conventions. |
| **Evaluation Helpers** | [script/run_many.py](script/run_many.py) sweeps over dozens of datasets, logging results to CSV + Neptune. |

---

## Model Zoo
Pre-trained checkpoints (stored in `ckpts/`, 2 MB each):

| Checkpoint | Training Mix |
|------------|--------------|
| **3KG**    | FB15k-237, WN18RR, CoDEx-M |
| **4HG**     | JF17K, WikiPeople, FB-AUTO, M-FB15K |
| **3KG+2HG**     | FB15k-237, WN18RR, CoDEx-M, JF17K, WikiPeople |

We also support **HCNet** inference in this repository. 


---

## Installation
HYPER requires **Python 3.9**, **PyTorch >= 2.1**, **PyG >= 2.4**.

### Conda
```bash
conda install pytorch=2.1.0 pytorch-cuda=11.8 cudatoolkit=11.8 \
              torch-scatter=2.1.2 pyg=2.4.0 \
              -c pytorch -c nvidia -c pyg -c conda-forge
conda install ninja easydict pyyaml -c conda-forge
```

### Pip
```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
            torch-geometric==2.4.0 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install ninja easydict pyyaml tqdm
```

> By default, the model will run with **Triton** for O(**V**) relational aggregation.  
> Ensure `nvcc` (CUDA 11.8+) is in `$CUDA_HOME` for GPU builds.

---

## Quick-Start
### Zero-Shot Inference (CPU)
```bash
python script/run.py \
    -c config/inference/HYPER_inference_inductive.yaml \
    --dataset FB15k237Inductive --version v1 \
    --epochs 0 --bpe null --gpus null \
    --ckpt ckpts/HYPER-3KG+2HG
```

### Zero-Shot Inference (GPU)
```bash
python script/run.py \
    -c config/inference/HYPER_inference_inductive.yaml \
    --dataset JF100 \
    --version None \
    --epochs 0 --bpe null --gpus [0] \
    --ckpt ckpts/HYPER-3KG+2HG
```

### Fine-Tuning
```bash
python script/run.py \
    -c config/inference/HYPER_inference_inductive.yaml \
    --dataset JF100  --version None --epochs 1 --bpe null --gpus [0] \
    --ckpt ckpts/HYPER-3KG+2HG \
    --finetune
```

### End-to-End Training
```bash
python script/run.py \
    -c config/inference/HYPER_inference_inductive.yaml \
    --dataset JF100  --version None --epochs 10 --bpe null --gpus [0] \
    --ckpt null \
    --train
```

---

## Running Experiments
### Single Dataset
* `script/run.py` – main entrypoint.  
  Arguments:
  | Flag | Meaning |
  |------|---------|
  | `-c` | YAML config |
  | `--dataset` | Dataset name |
  | `--version` | Inductive subset (if applicable) |
  | `--epochs`  | Training epochs (`0` = 0-shot) |
  | `--bpe`     | Batches per epoch (`null` = full) |
  | `--gpus`    | `null`, `[0]`, or `[0,1,…]` |
  | `--ckpt`    | Path to checkpoint (`null` = scratch) |
  | `--project` | Neptune project name |

### Many Datasets
* [script/run_many.py](script/run_many.py) – sequential benchmarking; emits `hyper_results_<timestamp>.csv`.

```bash
python script/run_many.py \
   -c /abs/path/config/inference/HYPER_inference_inductive.yaml \
   --gpus [0] \
   --ckpt /abs/path/ckpts/HYPER-3KG+2HG.pth \
   -d JF25,JF50,JF75,JF100 \
   --finetune
```

Arguments:  
`-d/--datasets` comma-sep list (`name[:version]`), `--finetune`, `--train`.


## Datasets
| Category | Example Names | Loader |
|----------|---------------|--------|
| **Transductive Hypergraphs** | JF17K, WikiPeople, FB-AUTO, M-FB15K | `hyper.datasets.<Name>` |
| **Inductive** | JF100, JF75, JF50, JF25, JFIND | `hyper.datasets.<Name>` |
| **Classic KGs** | FB15k-237, WN18RR, CoDEx* | `hyper.datasets.*` |
| **Custom** | Place inside `hypergraph_dataset/<DS_NAME>{/:version}`. |

Hypergraph tensors are padded to `max_arity` with 0s; first column stores relation id.

---

## Pre-Training
HYPER can be pre-trained on any mix of knowledge graphs and knowledge hypergraphs.

```bash
python script/pretrain.py \
   -c config/pretrain/pretrain_3KG+2HG.yaml \
   --gpus [0,1,2,3] --project Your/Project
```

*Configs live in `config/pretrain/`. Adjust `epochs`, `steps`, and graph list.*

---

## Directory Tour
```
hyper/                  core library
 ├── datasets.py        loaders & utility datasets
 ├── layers.py          HypergraphLayer + Triton rspmm
 ├── models.py          HYPER, RelHCNet, EntityHCNet
 ├── tasks.py           training / evaluation helpers
 └── util.py            logging, config, metrics, etc.
data_generation/        scripts to create & split datasets, and reification process
script/                 runnable experiment drivers
config/                 YAML configs (transductive, inductive, pretrain)
ckpts/                  pre-trained checkpoints
```

---


## License
This repository is licensed under the **MIT License**.

---

## Acknowledgements
Major parts of the training and evaluation engine are adapted from the ULTRA PyG implementation (https://github.com/DeepGraphLearning/ULTRA).  
We also thank the developers of **PyTorch**, **PyTorch-Geometric**, and **Triton** for their indispensable libraries.

Happy hyper-reasoning! 🚀

---
## Citation
If you find this repo or paper useful, please cite this:

```
@inproceedings{huang2026hyper,
  title={HYPER: A Foundation Model for Inductive Link Prediction with Knowledge Hypergraphs},
  author={Huang, Xingyue and Galkin, Mikhail and Bronstein, Michael M and Ceylan, {\.I}smail {\.I}lkan},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
