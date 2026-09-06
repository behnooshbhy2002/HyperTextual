"""
script/run_modular.py
=====================

Entry point for ModularHYPER — the single script that handles all four
combinations of (use_grel, use_text) via one yaml config.

Nothing in hyper/ or script/run.py is modified.
train_and_validate / test are imported directly from the original run.py.

Example runs
------------
# 1) original HYPER  (exact reproduction, no text)
python script/run_modular.py -c config/inference/HYPER_modular.yaml \
    --dataset JF25 --version null --epochs 10 --gpus null --bpe null --ckpt null \
    --use_grel true --use_text false

# 2) text-only baseline  (your new idea, no G_rel)
python script/run_modular.py -c config/inference/HYPER_modular.yaml \
    --dataset JF25 --version null --epochs 10 --gpus null --bpe null --ckpt null \
    --use_grel false --use_text true \
    --ent_text_json path/to/entity_text.json

# 3) full combined model
python script/run_modular.py -c config/inference/HYPER_modular.yaml \
    --dataset JF25 --version null --epochs 10 --gpus null --bpe null --ckpt null \
    --use_grel true --use_text true \
    --ent_text_json path/to/entity_text.json
"""

import os, sys, json, pprint, importlib.util
import torch
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hyper import util
from hyper_text.text_datasets import build_text_hypergraph_dataset
from hyper_text.text_models import ModularHYPER


# ── import train / test from the original run.py (unmodified) ─────────────────
def _load_run_fns():
    path = os.path.join(os.path.dirname(__file__), "run.py")
    spec = importlib.util.spec_from_file_location("_hyper_run", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train_and_validate, mod.test

train_and_validate, test = _load_run_fns()

SEP  = ">" * 30
LINE = "-" * 30


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
        logger.warning("Config: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    device    = util.get_device(cfg)
    use_grel  = bool(cfg.get("use_grel", True))
    use_text  = bool(cfg.get("use_text", True))
    task_name = cfg.task["name"]

    if util.get_rank() == 0:
        logger.warning(f"use_grel={use_grel}  use_text={use_text}")

    # ── text mappings (optional) ───────────────────────────────────────────────
    ent_raw2text = rel_raw2text = None
    if use_text and cfg.text.get("ent_raw2text_path"):
        with open(os.path.expanduser(cfg.text.ent_raw2text_path)) as f:
            ent_raw2text = json.load(f)
    if use_text and cfg.text.get("rel_raw2text_path"):
        with open(os.path.expanduser(cfg.text.rel_raw2text_path)) as f:
            rel_raw2text = json.load(f)

    # ── dataset ───────────────────────────────────────────────────────────────
    train_data, valid_data, test_data, dataset = build_text_hypergraph_dataset(
        root=os.path.expanduser(cfg.dataset.root),
        device=device,
        name=cfg.dataset["class"],
        ent_raw2text=ent_raw2text if use_text else None,
        rel_raw2text=rel_raw2text if use_text else None,
        text_model_name=cfg.text.model_name if use_text else None,
        text_cache_dir=(os.path.expanduser(cfg.text.cache_dir)
                        if use_text and cfg.text.get("cache_dir") else None),
    )
    for split in (train_data, valid_data, test_data):
        split.to(device)

    # ── model ─────────────────────────────────────────────────────────────────
    text_kwargs = {}
    if use_text:
        text_kwargs = dict(
            text_dim     = cfg.text.text_dim,
            fusion_type  = cfg.text.get("fusion_type", "gated"),
            text_dropout = cfg.text.get("text_dropout", 0.1),
        )

    model = ModularHYPER(
        rel_model_cfg    = cfg.model.relation_model,
        entity_model_cfg = cfg.model.entity_model,
        use_grel         = use_grel,
        use_text         = use_text,
        num_relations    = dataset.num_rel() if not use_grel else None,
        **text_kwargs,
    ).to(device)

    if cfg.get("checkpoint"):
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    # ── filtering graphs (identical to run.py) ────────────────────────────────
    if task_name == "InductiveInference":
        full_inf_edges  = torch.cat([test_data.edge_index,
                                     test_data.target_edge_index], dim=1)
        full_inf_etypes = torch.cat([test_data.edge_type,
                                     test_data.target_edge_type])
        test_filtered = Data(edge_index=full_inf_edges,
                             edge_type=full_inf_etypes,
                             num_nodes=test_data.num_nodes).to(device)
        val_filtered  = Data(
            edge_index=torch.cat([train_data.edge_index,
                                  valid_data.target_edge_index], dim=1),
            edge_type =torch.cat([train_data.edge_type,
                                  valid_data.target_edge_type]),
            num_nodes =valid_data.num_nodes,
        ).to(device)
    else:
        fd = Data(edge_index=dataset.dataclass["train_edge"],
                  edge_type =dataset.dataclass["train_rel"],
                  num_nodes =train_data.num_nodes).to(device)
        val_filtered = test_filtered = fd

     # <<< DEBUG: حذف کنید بعد از چک
    print(f"[DEBUG] dataset: {cfg.dataset['class']}")
    print(f"[DEBUG] train num_nodes: {train_data.num_nodes}")
    print(f"[DEBUG] train num_relations: {train_data.num_relations}")
    print(f"[DEBUG] max_arity: {train_data.max_arity}")
    # DEBUG >>>
    
    # ── train + eval ──────────────────────────────────────────────────────────
    train_and_validate(cfg, model, train_data, valid_data,
                       filtered_data=val_filtered, device=device,
                       batch_per_epoch=cfg.train.batch_per_epoch,
                       logger=logger, neptune_logger=neptune_logger)

    if util.get_rank() == 0:
        logger.warning(SEP)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=val_filtered, device=device,
         logger=logger, neptune_logger=neptune_logger, logger_mode="valid")

    if util.get_rank() == 0:
        logger.warning(SEP)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=test_filtered, device=device,
         logger=logger, neptune_logger=neptune_logger, logger_mode="test")