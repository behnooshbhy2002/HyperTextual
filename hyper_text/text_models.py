"""
text_models.py
==============

Modular HYPER extension with two independent on/off switches:

    use_grel : bool   (default True)
        True  → RelHCNet runs on G_rel (original paper idea).
                Entity init_feature gets the full relational context.
        False → G_rel is bypassed entirely.  A plain nn.Embedding table
                gives each relation a fixed learned vector (equivalent to
                the original paper's TransductiveHCNet, which the authors
                themselves included as a baseline).

    use_text : bool   (default True)
        True  → ent_text_emb / rel_text_emb are read from data and mixed
                into init_feature via TextFusion.
        False → no text signal anywhere; identical math to the original.

Four combinations:
    use_grel=True,  use_text=False  →  original HYPER (reproduced exactly)
    use_grel=False, use_text=True   →  text-only baseline (your new idea alone)
    use_grel=True,  use_text=True   →  full combined model
    use_grel=False, use_text=False  →  transductive embedding table only

All original hyper/ files remain untouched.
"""

import torch
from torch import nn
from torch.nn import functional as F

from hyper.util import static_positional_encoding
from hyper.models import RelHCNet, EntityHCNet
from hyper import tasks


# ──────────────────────────────────────────────
#  TextFusion  (unchanged from previous version)
# ──────────────────────────────────────────────

class TextFusion(nn.Module):
    """
    Combines a structural feature [B, N, model_dim] with a projected text
    feature [B, N, text_dim].

    fusion_type choices
    -------------------
    "additive"   : out = struct + proj(text)
    "gated"      : g = sigmoid(W[struct; proj(text)]);  out = g*struct + (1-g)*proj(text)
    "concat_mlp" : out = MLP([struct ; proj(text)])
    "film"       : out = struct * (1 + scale(proj(text))) + shift(proj(text))
    """

    def __init__(self, text_dim: int, model_dim: int,
                 fusion_type: str = "gated", dropout: float = 0.1):
        super().__init__()
        self.fusion_type = fusion_type
        self.proj = nn.Linear(text_dim, model_dim)
        self.text_dropout = nn.Dropout(dropout)

        if fusion_type == "gated":
            self.gate = nn.Linear(model_dim * 2, model_dim)
        elif fusion_type == "concat_mlp":
            self.mix = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim),
            )
        elif fusion_type == "film":
            self.film = nn.Linear(model_dim, model_dim * 2)
        elif fusion_type != "additive":
            raise ValueError(f"Unknown fusion_type: {fusion_type!r}")

    def forward(self, structural_feat, text_feat):
        text_proj = self.text_dropout(self.proj(text_feat))

        if self.fusion_type == "additive":
            return structural_feat + text_proj
        if self.fusion_type == "gated":
            g = torch.sigmoid(
                self.gate(torch.cat([structural_feat, text_proj], dim=-1))
            )
            return g * structural_feat + (1 - g) * text_proj
        if self.fusion_type == "concat_mlp":
            return self.mix(torch.cat([structural_feat, text_proj], dim=-1))
        if self.fusion_type == "film":
            scale, shift = self.film(text_proj).chunk(2, dim=-1)
            return structural_feat * (1 + scale) + shift
        raise ValueError(self.fusion_type)


# ──────────────────────────────────────────────
#  RelationModule  (wraps use_grel switch)
# ──────────────────────────────────────────────

class RelationModule(nn.Module):
    """
    Thin wrapper that either:
      - runs the full RelHCNet on G_rel  (use_grel=True), or
      - returns a learned embedding table lookup  (use_grel=False).

    Output shape is always [batch_size, num_relations, hidden_dim]
    so EntityHCNet can use it identically in both cases.

    use_text controls whether rel_text_emb seeds G_rel nodes (only
    meaningful when use_grel=True; ignored otherwise).
    """

    def __init__(self, rel_model_cfg: dict,
                 use_grel: bool = True,
                 use_text: bool = True,
                 text_dim: int | None = None,
                 fusion_type: str = "gated",
                 text_dropout: float = 0.1,
                 num_relations: int | None = None):
        super().__init__()
        self.use_grel = use_grel
        self.use_text = use_text

        if use_grel:
            self.rel_hcnet = RelHCNet(**rel_model_cfg)
            hidden_dim = rel_model_cfg["hidden_dims"][-1]
        else:
            # No G_rel: plain embedding table (= TransductiveHCNet's relation side)
            assert num_relations is not None, \
                "num_relations required when use_grel=False"
            hidden_dim = rel_model_cfg["hidden_dims"][-1]
            self.rel_embedding = nn.Embedding(num_relations, hidden_dim)

        # Text fusion on the relation side (only active when use_grel=True)
        if use_grel and use_text:
            assert text_dim is not None, "text_dim required when use_text=True"
            self.text_fusion = TextFusion(text_dim, rel_model_cfg["input_dim"],
                                          fusion_type=fusion_type,
                                          dropout=text_dropout)
        else:
            self.text_fusion = None

    # ------------------------------------------------------------------
    def forward(self, data, query_rel_idx):
        """
        Returns relation_representations : [batch_size, num_relations, hidden_dim]
        query_rel_idx : [batch_size] – the relation being queried (same for
                        all negatives in the batch, so we just use [0]).
        """
        if self.use_grel:
            return self._forward_grel(data, query_rel_idx)
        else:
            return self._forward_table(data, query_rel_idx)

    # --- G_rel path -------------------------------------------------------
    def _forward_grel(self, data, query_rel_idx):
        meta_graph = data.meta_graph

        if self.use_text and self.text_fusion is not None:
            # Inject text into G_rel node seeds before message passing
            return self._grel_with_text(meta_graph, query_rel_idx,
                                        getattr(data, "rel_text_emb", None))
        else:
            # Original RelHCNet.forward (unmodified call)
            return self.rel_hcnet(meta_graph, query=query_rel_idx)

    def _grel_with_text(self, meta_graph, query_rel_idx, rel_text_emb):
        """
        Mirrors TextRelHCNet.inference from the previous implementation,
        but now lives inside RelationModule so the flag is self-contained.
        """
        edge_list = torch.transpose(meta_graph.edge_index, 0, 1) + 1
        query = query_rel_idx + 1
        num_nodes = meta_graph.num_nodes + 1
        batch_size = len(query)

        # --- seed init_feature ---
        q_vec = torch.ones(batch_size, self.rel_hcnet.dims[0],
                           device=query.device, dtype=torch.float)
        idx = query.unsqueeze(-1).expand_as(q_vec)
        query_feature = torch.zeros(batch_size, num_nodes,
                                    self.rel_hcnet.dims[0], device=query.device)
        query_feature.scatter_add_(dim=1, index=idx.unsqueeze(1),
                                   src=q_vec.unsqueeze(1))
        init_feature = query_feature

        # >>> TEXT: seed every G_rel node with its text embedding
        if rel_text_emb is not None:
            padded = torch.cat(
                [torch.zeros(1, rel_text_emb.size(-1),
                             device=rel_text_emb.device), rel_text_emb], dim=0
            )                                               # (num_rel+1, text_dim)
            text_feat = padded.unsqueeze(0).expand(batch_size, -1, -1)
            init_feature = self.text_fusion(init_feature, text_feat)
        # <<< TEXT

        init_feature[:, self.rel_hcnet.padding_idx, :] = 0  # clear padding

        layer_input = init_feature
        for layer in self.rel_hcnet.layers:
            hidden = F.relu(layer(layer_input, edge_list,
                                  meta_graph.edge_type, init_feature))
            if self.rel_hcnet.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        score = self.rel_hcnet.mlp(layer_input)
        return score[:, 1:, :]   # drop the padding node (index 0)

    # --- table path -------------------------------------------------------
    def _forward_table(self, data, query_rel_idx):
        # Returns a dummy [batch_size, num_relations, dim] tensor by
        # broadcasting the single-query embedding.
        # EntityHCNet uses:
        #   relation_representations[arange(batch_size), r_idx[:, 0]]
        # so it only ever reads the query row – we fill every row with the
        # query embedding (cheap, correct for what EntityHCNet actually reads).
        batch_size = query_rel_idx.size(0)
        num_rel, dim = self.rel_embedding.weight.shape
        emb = self.rel_embedding(query_rel_idx)        # [batch_size, dim]
        return emb.unsqueeze(1).expand(-1, num_rel, -1)  # [batch_size, num_rel, dim]


# ──────────────────────────────────────────────
#  ModularEntityHCNet  (wraps use_text switch)
# ──────────────────────────────────────────────

class ModularEntityHCNet(EntityHCNet):
    """
    Subclass of EntityHCNet that optionally injects entity text embeddings
    into init_feature.  When use_text=False the behaviour is bit-identical
    to the unmodified EntityHCNet.

    The two added lines in inference() are clearly marked with
        # >>> TEXT ...  # <<< TEXT
    so they are easy to compare against the parent.
    """

    def __init__(self, *args,
                 use_text: bool = True,
                 text_dim: int | None = None,
                 fusion_type: str = "gated",
                 text_dropout: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_text = use_text

        if use_text:
            assert text_dim is not None, "text_dim required when use_text=True"
            self.text_fusion = TextFusion(text_dim, self.dims[0],
                                          fusion_type=fusion_type,
                                          dropout=text_dropout)
        else:
            self.text_fusion = None

    # ------ overridden inference (parent copy + two marked TEXT blocks) --------
    def inference(self, r_idx, entities_idx, edge_list, rel_list,
                  num_nodes, relation_representations=None,
                  ent_text_emb=None):

        r_idx = r_idx.t()
        entities_idx = entities_idx.permute(1, 0, 2)
        edge_list = edge_list.t()

        max_arity = len(edge_list[0])
        static_encodings = static_positional_encoding(max_arity + 1, self.input_dim)
        self.position = static_encodings.to(r_idx.device)
        self.position[self.padding_idx] = torch.ones(self.input_dim)

        arity = torch.ones_like(r_idx) * entities_idx.size(-1)
        batch_size = len(r_idx)

        entities_idx = torch.einsum("ijk->ikj", entities_idx)
        all_idx = entities_idx
        mask_for_diff = torch.all(
            all_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, all_idx.size(-1)) == all_idx,
            dim=-1,
        )
        pos_index_to_search = (mask_for_diff == False).int().argmax(dim=1)
        assert torch.all(r_idx[:, 0].unsqueeze(-1).expand(-1, r_idx.size(1)) == r_idx)
        assert torch.all(torch.sum(mask_for_diff, dim=-1) >= all_idx.size(1) - 1)
        assert torch.all(arity[:, 0].unsqueeze(-1).expand(-1, arity.size(1)) == arity)
        assert torch.all(pos_index_to_search <= arity[:, 0])

        if not self.transductive_hcnet:
            query = relation_representations[
                torch.arange(batch_size, device=r_idx.device), r_idx[:, 0]
            ]
        else:
            query = self.query(r_idx[:, 0])

        init_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)
        result_tensor = torch.ones((batch_size, max_arity), dtype=torch.int, device=r_idx.device)
        range_tensor = torch.arange(max_arity, device=result_tensor.device).expand(batch_size, max_arity)
        arity_range = arity[:, 0].unsqueeze(1).expand(-1, max_arity)

        mask = range_tensor < arity_range
        result_tensor *= mask
        zero_out_mask = range_tensor == pos_index_to_search.unsqueeze(1)
        result_tensor[zero_out_mask] = 0

        index_arity_without_self = (all_idx[:, :, 0] * result_tensor
                                    ).unsqueeze(-1).expand(-1, -1, self.dims[0]).to(torch.int64)

        # query feature
        query_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)
        query_feature.scatter_add_(dim=1, index=index_arity_without_self,
                                   src=query.unsqueeze(1).expand(-1, max_arity, -1))
        init_feature += query_feature

        # positional feature
        pos_src_index = result_tensor * torch.arange(
            1, max_arity + 1, device=result_tensor.device
        ).expand(batch_size, max_arity)
        pos_src = self.position[pos_src_index]
        pos_init_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)
        pos_init_feature.scatter_add_(dim=1, index=index_arity_without_self, src=pos_src)
        init_feature += pos_init_feature

        # >>> TEXT: entity text signal (skipped entirely when use_text=False)
        if self.use_text and self.text_fusion is not None and ent_text_emb is not None:
            text_dim = ent_text_emb.size(-1)
            entity_id_per_slot = all_idx[:, :, 0] * result_tensor
            node_text = ent_text_emb[entity_id_per_slot]
            text_index = (index_arity_without_self[..., :1]
                          .expand(-1, -1, text_dim))
            text_feature = torch.zeros(batch_size, num_nodes, text_dim, device=r_idx.device)
            text_feature.scatter_add_(dim=1, index=text_index, src=node_text)
            init_feature = self.text_fusion(init_feature, text_feature)
        # <<< TEXT

        init_feature[:, 0, :] = 0

        layer_input = init_feature
        for layer in self.layers:
            hidden = F.relu(
                layer(layer_input, edge_list, rel_list, init_feature, relation_representations)
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        output = layer_input

        output = torch.cat(
            [output, query.unsqueeze(1).expand(-1, output.size(1), -1)], dim=-1
        )

        in_batch_tensor = all_idx * (
            torch.logical_not(mask_for_diff).int()
            .unsqueeze(-1).expand(-1, -1, all_idx.size(-1))
        )
        collapsed_tensor = in_batch_tensor[torch.any(in_batch_tensor != 0, dim=2)]
        feature = output.gather(
            1, collapsed_tensor.unsqueeze(-1).expand(-1, -1, output.size(-1))
        )
        score = self.mlp(feature).squeeze(-1)
        return score

    # ------ overridden forward (routes text emb through) ------------------
    def forward(self, data, batch, relation_representations, use_grel: bool):
        r_idx, entities_idx = batch[-1].T, batch[:-1].T
        edge_list, rel_list, num_nodes = data.edge_index.T, data.edge_type, data.num_nodes

        r_idx = r_idx.to(edge_list.device)
        entities_idx = entities_idx.to(edge_list.device)

        ent_text_emb = getattr(data, "ent_text_emb", None) if self.use_text else None

        if self.training:
            edge_list, rel_list = self.remove_easy_edge(
                r_idx, entities_idx, edge_list, rel_list
            )
            data = tasks.build_weighted_graph(data)
            # re-attach custom attrs after build_weighted_graph
            if ent_text_emb is not None:
                data.ent_text_emb = ent_text_emb

        edge_list = edge_list.T
        score = self.inference(
            r_idx, entities_idx, edge_list, rel_list, num_nodes,
            relation_representations,
            ent_text_emb=ent_text_emb,
        )
        return score


# ──────────────────────────────────────────────
#  ModularHYPER  — the single entry point
# ──────────────────────────────────────────────

class ModularHYPER(nn.Module):
    """
    Drop-in replacement for both HYPER and TextHYPER.

    Config keys (all optional; defaults reproduce the original paper):
    ─────────────────────────────────────────────────────────────────
    use_grel  : bool   default True   — enable G_rel + RelHCNet
    use_text  : bool   default True   — enable text embeddings

    rel_model_cfg    : dict  — same as original HYPER config
    entity_model_cfg : dict  — same as original HYPER config
    text_dim         : int   — embedding dim from SentenceTransformer (e.g. 384)
    fusion_type      : str   — "gated" | "additive" | "concat_mlp" | "film"
    text_dropout     : float — default 0.1
    num_relations    : int   — required only when use_grel=False
    """

    def __init__(self,
                 rel_model_cfg: dict,
                 entity_model_cfg: dict,
                 use_grel: bool = True,
                 use_text: bool = True,
                 text_dim: int | None = None,
                 fusion_type: str = "gated",
                 text_dropout: float = 0.1,
                 num_relations: int | None = None):
        super().__init__()
        self.use_grel = use_grel
        self.use_text = use_text

        # --- relation side ---
        self.relation_module = RelationModule(
            rel_model_cfg=rel_model_cfg,
            use_grel=use_grel,
            use_text=use_text,
            text_dim=text_dim,
            fusion_type=fusion_type,
            text_dropout=text_dropout,
            num_relations=num_relations,
        )

        # --- entity side ---
        self.entity_model = ModularEntityHCNet(
            **entity_model_cfg,
            use_text=use_text,
            text_dim=text_dim,
            fusion_type=fusion_type,
            text_dropout=text_dropout,
        )

    def forward(self, data, batch):
        r_idx = batch[-1].T          # [batch_size, num_neg+1]

        # relation representations — shape [batch_size, num_rel, dim]
        rel_repr = self.relation_module(data, query_rel_idx=r_idx[0])

        score = self.entity_model(
            data, batch, rel_repr, use_grel=self.use_grel
        )
        return score