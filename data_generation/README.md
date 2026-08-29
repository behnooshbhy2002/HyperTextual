# Data Generation

This directory provides scripts to:

1. **Reify** a hypergraph into a binary knowledge graph (KG).
2. **Generate** node-and-relation-inductive link prediction datasets from hypergraphs.

---

## Prerequisites

- Python 3.9+
- Dependencies:
  ```bash
  pip install networkx numpy tqdm 
  ```

## 1. Reify Hypergraph to Knowledge Graph

Use **reify_hypergraph.py** to convert each hyper-edge into a new entity plus binary relations:

```bash
python reify_hypergraph.py \
  --input_dir <HYPERGRAPH_DIR> \
  --output_dir <BINARY_KG_DIR> \
  [--analyze_only]
```

- **Input**: folder with `train.txt`, `valid.txt`, `test.txt`, etc.
- **Output**: same splits in binary format under `<BINARY_KG_DIR>`.
- `--analyze_only`: only prints statistics, does not write files.

---

## 2. Generate Inductive Link Prediction Datasets

### 2.1 Split hypergraph for training & inference

Use **generate_dataset.py** to sample relation splits and two-hop neighborhoods:

```bash
python generate_dataset.py \
  --data_src <SOURCE_SPLIT> \
  --data_tgt <TARGET_DIR> \
  --n_train <#train_seeds> \
  --n_test <#test_seeds> \
  --p_rel <relation_split_ratio> \
  --p_tri <merge_ratio> \
  [--percentage_binary_remove <0.0>] \
  [--seed <int>] [--no_save]
```

- **Outputs** in `data_generation/<TARGET_DIR>/`:
  - `train.txt`: training hyper-edges
  - `hypergraph_inference.txt`: inference hyper-edges for eval

### 2.2 Create validation & test splits

Use **val_test.py** to extract validation and test sets from the inference hypergraph via a spanning hypertree:

```bash
python val_test.py \
  --data <TARGET_DIR> [--seed <int>] [--no_save]
```

- Reads `data_generation/<TARGET_DIR>/hypergraph_inference.txt`
- Writes `aux.txt`, `valid.txt`, `test.txt` into the same folder.

---

## Utility Functions

All core logic is in **utils.py**:
- Tuple reading & deduplication
- Neighbor sampling (`sample_2hop`)
- Connected component & spanning hypertree
- Binary relation removal
