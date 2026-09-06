"""
Entry point for training/evaluating TextHYPER (HYPER + entity/relation text
semantics). This file is a NEW file -- `script/run.py` is never modified.

It reuses, unmodified, from the original repo:
  - hyper.util.parse_args / load_config / create_working_directory / get_device / get_root_logger
  - script.run.train_and_validate / script.run.test   (loaded dynamically below,
    since they are 100% generic w.r.t. the model: they just call
    `model(data, batch)`, which TextHYPER also implements)

It uses, from the new hyper_text package:
  - hyper_text.text_datasets.build_text_hypergraph_dataset  (loads data + text embeds)
  - hyper_text.text_models.TextHYPER

Usage (mirrors the original run.py):
    python script/run_text.py -c config/inference/HYPER_text_inductive.yaml \
        --dataset JF25 --version null --epochs 10 --gpus null --bpe null --ckpt null \
        --ent_text_json path/to/entity_text.json --rel_text_json path/to/relation_text.json
"""

import os
import sys
import json
import pprint
import importlib.util

import torch
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hyper import util
from hyper_text.text_datasets import build_text_hypergraph_dataset
from hyper_text.text_models import TextHYPER


def _load_run_py_functions():
    """
    Dynamically loads `script/run.py` as a module (WITHOUT executing its
    `if __name__ == "__main__":` block, since that only runs when the file
    itself is the entry point) so we can reuse `train_and_validate` and
    `test` verbatim instead of duplicating the training loop.
    """
    run_py_path = os.path.join(os.path.dirname(__file__), "run.py")
    spec = importlib.util.spec_from_file_location("hyper_original_run", run_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train_and_validate, module.test


train_and_validate, test = _load_run_py_functions()

separator = ">" * 30
line = "-" * 30


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    neptune_logger = None
    if util.get_rank() == 0:
        neptune_logger = util.create_neptune_run(args, vars, cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    task_name = cfg.task["name"]
    device = util.get_device(cfg)

    # ---- load raw-id -> text mappings (optional) ------------------------
    ent_raw2text, rel_raw2text = None, None
    if cfg.text.get("ent_raw2text_path"):
        with open(os.path.expanduser(cfg.text.ent_raw2text_path)) as f:
            ent_raw2text = json.load(f)
    if cfg.text.get("rel_raw2text_path"):
        with open(os.path.expanduser(cfg.text.rel_raw2text_path)) as f:
            rel_raw2text = json.load(f)

    # ---- load dataset + attach text embeddings ---------------------------
    train_data, valid_data, test_data, dataset = build_text_hypergraph_dataset(
        root=os.path.expanduser(cfg.dataset.root),
        device=device,
        name=cfg.dataset["class"],
        ent_raw2text=ent_raw2text,
        rel_raw2text=rel_raw2text,
        text_model_name=cfg.text.model_name,
        text_cache_dir=os.path.expanduser(cfg.text.cache_dir) if cfg.text.get("cache_dir") else None,
        inductive_dataset=True,
        alternative_build=True,
    )
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    # ---- build model -------------------------------------------------
    assert cfg.model["class"] == "TextHYPER", "run_text.py only supports model.class: TextHYPER"
    model = TextHYPER(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)

    # ---- filtering graphs (identical logic to script/run.py) ------------
    if task_name == "InductiveInference":
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class'] or cfg.dataset['class'] in [
            ds + version for version in ["100", "75", "50", "25"] for ds in ["JF", "WP", "MFB"]
        ]:
            full_inference_edges = torch.cat(
                [valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1
            )
            full_inference_etypes = torch.cat(
                [valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type]
            )
            test_filtered_data = Data(
                edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes
            )
            val_filtered_data = test_filtered_data
        else:
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(
                edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes
            )
            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type]),
                num_nodes=valid_data.num_nodes,
            )
    else:
        filtered_data = Data(
            edge_index=dataset.dataclass["train_edge"], edge_type=dataset.dataclass["train_rel"],
            num_nodes=train_data.num_nodes,
        )
        val_filtered_data = test_filtered_data = filtered_data

    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)

    train_and_validate(
        cfg, model, train_data, valid_data, filtered_data=val_filtered_data,
        device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger, neptune_logger=neptune_logger,
    )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger,
         neptune_logger=neptune_logger, logger_mode="valid")
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger,
         neptune_logger=neptune_logger, logger_mode="test")
