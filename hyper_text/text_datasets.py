"""
Loads a HYPER dataset EXACTLY the way `hyper.datasets.build_hypergraph_dataset`
does (we call the very same, unmodified classes/functions:
`Hypergraph_Dataset` and `build_weighted_graph`), but:

  1) keeps a handle on `dataset.ent2id` / `dataset.rel2id`, which the
     original `build_hypergraph_dataset` throws away, so we can build a
     text-embedding lookup table indexed the same way the model indexes
     entities/relations, and

  2) attaches the resulting `ent_text_emb` / `rel_text_emb` tensors onto the
     `train_data` / `valid_data` / `test_data` objects, so `TextHYPER` can
     just read `data.ent_text_emb` / `data.rel_text_emb` at forward time.

Nothing in `hyper/` is modified or monkey-patched; we simply call it.
"""

import torch
from torch_geometric.data import Data

from hyper.datasets import Hypergraph_Dataset, build_weighted_graph
from hyper_text.text_encoder import build_text_lookup


def _ids_to_texts(id2str: dict, raw2text: dict, default: str = "") -> dict:
    """
    id2str:   {0: "", 1: "m.02mjmr", 2: "m.09c7w0", ...}   (dataset.ent2id inverted)
    raw2text: {"m.02mjmr": "Barack Obama, 44th U.S. President", ...}  (your own mapping)
    -> {0: "", 1: "Barack Obama, 44th U.S. President", 2: default, ...}
    """
    return {i: raw2text.get(raw, default) for i, raw in id2str.items()}


def build_text_hypergraph_dataset(
    root,
    device,
    name,
    ent_raw2text: dict | None = None,
    rel_raw2text: dict | None = None,
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    text_cache_dir: str | None = None,
    inductive_dataset: bool = True,
    alternative_build: bool = True,
    binary_dataset: bool = False,
):
    """
    Drop-in analogue of `hyper.datasets.build_hypergraph_dataset`, but also
    returns entity/relation text embeddings attached to each split.

    ent_raw2text / rel_raw2text : dict[str, str], optional
        Maps the RAW id used in train.txt/valid.txt/test.txt (e.g. a
        Freebase mid, or already a human-readable name) to a text
        description. If None, the raw id string itself is used as the
        "text" (works out of the box for datasets whose raw ids are
        already readable names; for cryptic-id datasets you should supply
        a real mapping for this to be useful).

    Returns
    -------
    (train_data, valid_data, test_data, dataset)
        dataset is the underlying `Hypergraph_Dataset`, kept around so you
        can still access `dataset.ent2id` / `dataset.rel2id` /
        `dataset.max_arity` etc. if needed.
    """
    dataset = Hypergraph_Dataset(
        name, device=device, root=root,
        binary_dataset=binary_dataset, inductive_dataset=inductive_dataset,
    )

    train_data = Data(
        edge_index=dataset.dataclass["train_edge_graph"], edge_type=dataset.dataclass["train_rel_graph"],
        num_nodes=dataset.num_ent(),
        target_edge_index=dataset.dataclass["train_edge"], target_edge_type=dataset.dataclass["train_rel"],
        num_relations=dataset.num_rel(), max_arity=dataset.max_arity, device=device,
    )
    if alternative_build:
        valid_data = Data(
            edge_index=dataset.dataclass["test_edge_graph"], edge_type=dataset.dataclass["test_rel_graph"],
            num_nodes=dataset.num_ent(),
            target_edge_index=dataset.dataclass["valid_edge"], target_edge_type=dataset.dataclass["valid_rel"],
            num_relations=dataset.num_rel(), max_arity=dataset.max_arity, device=device,
        )
    else:
        valid_data = Data(
            edge_index=dataset.dataclass["train_edge_graph"], edge_type=dataset.dataclass["train_rel_graph"],
            num_nodes=dataset.num_ent(),
            target_edge_index=dataset.dataclass["valid_edge"], target_edge_type=dataset.dataclass["valid_rel"],
            num_relations=dataset.num_rel(), max_arity=dataset.max_arity, device=device,
        )
    test_data = Data(
        edge_index=dataset.dataclass["test_edge_graph"], edge_type=dataset.dataclass["test_rel_graph"],
        num_nodes=dataset.num_ent(),
        target_edge_index=dataset.dataclass["test_edge"], target_edge_type=dataset.dataclass["test_rel"],
        num_relations=dataset.num_rel(), max_arity=dataset.max_arity, device=device,
    )

    # build the relation hypergraph (G_rel) -- unmodified original function
    train_data = build_weighted_graph(train_data)
    valid_data = build_weighted_graph(valid_data)
    test_data = build_weighted_graph(test_data)

    # ---- build text lookup tables (skipped when text_model_name is None) ------
    if text_model_name is not None:
        id2ent_raw = {v: k for k, v in dataset.ent2id.items()}
        id2rel_raw = {v: k for k, v in dataset.rel2id.items()}

        ent_raw2text = ent_raw2text or {}
        rel_raw2text = rel_raw2text or {}

        ent_id2text = _ids_to_texts(id2ent_raw, ent_raw2text, default="")
        for i, raw in id2ent_raw.items():
            if i != 0 and i not in ent_raw2text and raw not in ent_raw2text:
                ent_id2text[i] = raw
        rel_id2text = _ids_to_texts(id2rel_raw, rel_raw2text, default="")
        for i, raw in id2rel_raw.items():
            if raw not in rel_raw2text:
                rel_id2text[i] = raw

        ent_cache = f"{text_cache_dir}/{name}_entity_text.pt" if text_cache_dir else None
        rel_cache = f"{text_cache_dir}/{name}_relation_text.pt" if text_cache_dir else None

        ent_text_emb = build_text_lookup(ent_id2text, model_name=text_model_name, cache_path=ent_cache).to(device)
        rel_text_emb = build_text_lookup(rel_id2text, model_name=text_model_name, cache_path=rel_cache).to(device)

        for split in (train_data, valid_data, test_data):
            split.ent_text_emb = ent_text_emb
            split.rel_text_emb = rel_text_emb
    # when text_model_name is None, no ent_text_emb/rel_text_emb is attached;
    # ModularEntityHCNet checks use_text=False and never reads them.

    return train_data, valid_data, test_data, dataset


# Convenience wrappers mirroring hyper.datasets.JF25 / WPIND / etc.
def JF25_with_text(root, device, ent_raw2text=None, rel_raw2text=None, **kwargs):
    return build_text_hypergraph_dataset(
        root, device, name="JF-25",
        ent_raw2text=ent_raw2text, rel_raw2text=rel_raw2text,
        inductive_dataset=True, alternative_build=True, **kwargs,
    )