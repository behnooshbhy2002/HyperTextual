"""
Builds a lookup table  id -> semantic_embedding  for the entities or
relations of a HYPER dataset.

The only thing this module needs from the outside is a dict:
    id2text = {0: "", 1: "Barack Obama", 2: "United States", ...}
(id 0 is always the padding entry in HYPER; it always gets a zero vector.)

You are free to fill `id2text` however you like:
  - dataset already has human-readable names           -> use them directly
  - dataset only has opaque ids (Freebase mids, etc.)   -> map mid -> label
    with an external table (e.g. Freebase mid2name, Wikidata labels) and,
    optionally, ask an LLM to turn short cryptic labels into 1-2 sentence
    descriptions for a richer signal.

Usage
-----
    from hyper_text.text_encoder import build_text_lookup

    ent_text_emb = build_text_lookup(
        id2text=ent_id2text,
        cache_path="cache/JF25_entity_text.pt",
    )
"""

import os
import json
import torch


_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sentence_transformer(model_name, device):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for build_text_lookup(). "
            "Install it with: pip install sentence-transformers --break-system-packages"
        ) from e
    return SentenceTransformer(model_name, device=device)


@torch.no_grad()
def build_text_lookup(
    id2text: dict,
    model_name: str = _DEFAULT_MODEL_NAME,
    cache_path: str | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    encoder_fn=None,
):
    """
    Parameters
    ----------
    id2text : dict[int, str]
        Maps every entity/relation id (0..N-1, as used by `Hypergraph_Dataset`)
        to a short text description. Missing ids default to "".
    model_name : str
        Any sentence-transformers model name. Ignored if `encoder_fn` is given.
    cache_path : str, optional
        If given and the file exists, the cached tensor is loaded directly
        (huge speedup on the 2nd+ run). If given and the file does not
        exist, the computed tensor is saved there.
    device : str
        Device used to *compute* the embeddings (the returned tensor is
        always moved to cpu; move it to the training device yourself, since
        it must live on `data.ent_text_emb` / `data.rel_text_emb`).
    encoder_fn : callable(list[str]) -> torch.Tensor, optional
        Escape hatch: pass your own encoder (e.g. a call to an API, a
        cached BERT model, a multilingual model, ...) instead of
        sentence-transformers. Must return a (len(texts), text_dim) tensor.

    Returns
    -------
    torch.Tensor of shape (num_ids, text_dim), float32, on cpu.
    id 0 (padding) is always all-zeros regardless of id2text[0].
    """
    if cache_path is not None and os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    num_ids = max(id2text.keys()) + 1
    texts = [id2text.get(i, "") for i in range(num_ids)]

    if encoder_fn is not None:
        embeddings = encoder_fn(texts)
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
    else:
        model = _load_sentence_transformer(model_name, device)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=device,
        ).float()

    embeddings = embeddings.cpu()
    embeddings[0] = 0.0  # padding id always maps to the zero vector

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(embeddings, cache_path)

    return embeddings


def load_id2text_json(path: str) -> dict:
    """
    Small helper to load a { "raw_name_or_mid": "text description", ... }
    json file and you still need to remap it to {int_id: text} using
    dataset.ent2id / dataset.rel2id -- see `text_datasets.py::_ids_to_texts`.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
