"""
Text-aware variants of the HYPER model.

Design
------
`hyper/models.py` is NEVER imported-and-mutated or monkey-patched: we only
`import` its classes and *subclass* them. `RelHCNet.inference` and
`EntityHCNet.inference` mix positional/query info into `init_feature` in a
few tightly-coupled lines, so a clean override means literally copying the
parent method and adding the new lines -- every added block is wrapped in
    # >>> TEXT ...
    # <<< TEXT
so that if you (or the authors) update `hyper/models.py`, you can diff the
two `inference` methods side by side and re-apply just the marked blocks.

Where the text signal is injected
----------------------------------
1. TextRelHCNet: every relation-node in the meta-graph (G_rel) is seeded
   with its own text embedding instead of zero (previously only the query
   relation got a non-zero seed -- all-ones -- everyone else started blind).
2. TextEntityHCNet: on top of the existing query+positional init_feature,
   every entity slot in the hyperedge is also seeded with that entity's
   text embedding.

Both injections go through a small `TextFusion` module so you can switch
between additive / gated / concat-mlp / FiLM combination via config,
without touching the surrounding message-passing code at all.
"""

import torch
from torch import nn
from torch.nn import functional as F

from hyper.util import static_positional_encoding
from hyper.models import RelHCNet, EntityHCNet
from hyper import tasks


class TextFusion(nn.Module):
    """
    Combines a structural feature vector (from HYPER's own init_feature)
    with a projected text-embedding vector.

    fusion_type:
      - "additive"   : struct + proj(text)                      (simplest)
      - "concat_mlp" : MLP([struct ; proj(text)])                (most capacity)
      - "gated"      : g*struct + (1-g)*proj(text), g learned    (recommended default)
      - "film"       : struct * (1+scale(text)) + shift(text)    (feature-wise modulation)

    `dropout` randomly zeroes the *text* branch during training. This keeps
    the model from over-relying on text and collapsing to poor behavior for
    entities/relations you don't have good text for (or didn't have text
    for at train time) -- a bit like word-dropout for the semantic channel.
    """

    def __init__(self, text_dim, model_dim, fusion_type="gated", dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        self.proj = nn.Linear(text_dim, model_dim)
        self.text_dropout = nn.Dropout(dropout)

        if fusion_type == "gated":
            self.gate = nn.Linear(model_dim * 2, model_dim)
        elif fusion_type == "concat_mlp":
            self.mix = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)
            )
        elif fusion_type == "film":
            self.film = nn.Linear(model_dim, model_dim * 2)
        elif fusion_type != "additive":
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, structural_feat, text_feat):
        text_proj = self.text_dropout(self.proj(text_feat))

        if self.fusion_type == "additive":
            return structural_feat + text_proj
        if self.fusion_type == "concat_mlp":
            return self.mix(torch.cat([structural_feat, text_proj], dim=-1))
        if self.fusion_type == "gated":
            g = torch.sigmoid(self.gate(torch.cat([structural_feat, text_proj], dim=-1)))
            return g * structural_feat + (1 - g) * text_proj
        if self.fusion_type == "film":
            scale, shift = self.film(text_proj).chunk(2, dim=-1)
            return structural_feat * (1 + scale) + shift
        raise ValueError(self.fusion_type)


class TextRelHCNet(RelHCNet):
    """
    Same as `hyper.models.RelHCNet`, except every relation node in the
    meta-graph gets seeded with its text embedding (not just the query
    relation with the original all-ones vector).
    """

    def __init__(self, *args, text_dim=None, fusion_type="gated", text_dropout=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert text_dim is not None, "TextRelHCNet requires text_dim"
        self.text_fusion = TextFusion(text_dim, self.dims[0], fusion_type=fusion_type, dropout=text_dropout)

    # ---- overridden copy of RelHCNet.inference --------------------------
    def inference(self, query_idx, edge_list, rel_list, num_nodes, rel_text_emb=None):
        batch_size = len(query_idx)

        query = torch.ones(query_idx.shape[0], self.dims[0], device=query_idx.device, dtype=torch.float)
        index = query_idx.unsqueeze(-1).expand_as(query)
        query_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=query_idx.device)
        query_feature.scatter_add_(dim=1, index=index.unsqueeze(1), src=query.unsqueeze(1))

        init_feature = query_feature

        # >>> TEXT: seed EVERY relation node with its text embedding, not just
        # the query relation. `rel_text_emb` is (num_relations, text_dim), we
        # shift it by one padding row to line up with the +1-shifted node ids
        # used inside this class (see forward() below).
        if rel_text_emb is not None:
            padded_text = torch.cat(
                [torch.zeros(1, rel_text_emb.size(-1), device=rel_text_emb.device), rel_text_emb], dim=0
            )  # (num_relations + 1, text_dim), row 0 = padding
            text_feat = padded_text.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_nodes, text_dim)
            init_feature = self.text_fusion(init_feature, text_feat)
        # <<< TEXT

        init_feature[:, self.padding_idx, :] = 0  # clear the padding node

        layer_input = init_feature
        for layer in self.layers:
            hidden = F.relu(layer(layer_input, edge_list, rel_list, init_feature))
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        output = layer_input

        score = self.mlp(output)
        return score

    # ---- overridden copy of RelHCNet.forward (only to pass rel_text_emb through)
    def forward(self, relation_hypergraph, query, rel_text_emb=None):
        edge_list, rel_list, num_nodes = (
            relation_hypergraph.edge_index, relation_hypergraph.edge_type, relation_hypergraph.num_nodes
        )
        edge_list = torch.transpose(edge_list, 0, 1) + 1
        query = query + 1
        num_nodes = num_nodes + 1

        relation_feature = self.inference(query, edge_list, rel_list, num_nodes, rel_text_emb=rel_text_emb)
        query = query - 1
        return relation_feature[:, 1:, :]


class TextEntityHCNet(EntityHCNet):
    """
    Same as `hyper.models.EntityHCNet`, except every entity slot in the
    hyperedge is additionally seeded with that entity's text embedding, on
    top of the existing query+positional init_feature.
    """

    def __init__(self, *args, text_dim=None, fusion_type="gated", text_dropout=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert text_dim is not None, "TextEntityHCNet requires text_dim"
        self.text_fusion = TextFusion(text_dim, self.dims[0], fusion_type=fusion_type, dropout=text_dropout)

    # ---- overridden copy of EntityHCNet.inference ------------------------
    def inference(self, r_idx, entities_idx, edge_list, rel_list, num_nodes,
                  relation_representations=None, ent_text_emb=None):
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
            all_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, all_idx.size(-1)) == all_idx, dim=-1
        )
        pos_index_to_search = (mask_for_diff == False).int().argmax(dim=1)
        assert torch.all(r_idx[:, 0].unsqueeze(-1).expand(-1, r_idx.size(1)) == r_idx)
        assert torch.all(torch.sum(mask_for_diff, dim=-1) >= all_idx.size(1) - 1)
        assert torch.all(arity[:, 0].unsqueeze(-1).expand(-1, arity.size(1)) == arity)
        assert torch.all(pos_index_to_search <= arity[:, 0])

        if not self.transductive_hcnet:
            query = relation_representations[torch.arange(batch_size, device=r_idx.device), r_idx[:, 0]]
        else:
            assert self.query is not None
            query = self.query(r_idx[:, 0])

        init_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)
        result_tensor = torch.ones((batch_size, max_arity), dtype=torch.int, device=r_idx.device)
        range_tensor = torch.arange(max_arity, device=result_tensor.device).expand(batch_size, max_arity)
        arity_range = arity[:, 0].unsqueeze(1).expand(-1, max_arity)

        mask = range_tensor < arity_range
        result_tensor *= mask
        zero_out_mask = range_tensor == pos_index_to_search.unsqueeze(1)
        result_tensor[zero_out_mask] = 0

        index_arity_without_self = all_idx[:, :, 0] * result_tensor
        index_arity_without_self = index_arity_without_self.unsqueeze(-1).expand(-1, -1, self.dims[0]).to(torch.int64)

        query_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)
        query_feature.scatter_add_(
            dim=1, index=index_arity_without_self, src=query.unsqueeze(1).expand(-1, max_arity, -1)
        )
        init_feature += query_feature

        pos_src_index = result_tensor * torch.arange(1, max_arity + 1, device=result_tensor.device).expand(batch_size, max_arity)
        pos_src = self.position[pos_src_index]
        pos_init_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)
        pos_init_feature.scatter_add_(dim=1, index=index_arity_without_self, src=pos_src)
        init_feature += pos_init_feature

        # >>> TEXT: seed every entity slot with its own text embedding too.
        # `ent_text_emb` is indexed by the SAME entity ids used everywhere
        # else in HYPER (id 0 = padding -> zero vector already guaranteed by
        # build_text_lookup()), so no extra shifting is needed here (unlike
        # the relation side, entity ids are not re-based inside this class).
        if ent_text_emb is not None:
            text_dim = ent_text_emb.size(-1)
            # entity id for each (batch, arity-slot), 0 for the masked-out slot(s)
            entity_id_per_slot = all_idx[:, :, 0] * result_tensor           # (B, max_arity)
            node_text = ent_text_emb[entity_id_per_slot]                    # (B, max_arity, text_dim)
            text_index = index_arity_without_self[..., :1].expand(-1, -1, text_dim)  # reuses the same node ids
            text_feature = torch.zeros(batch_size, num_nodes, text_dim, device=r_idx.device)
            text_feature.scatter_add_(dim=1, index=text_index, src=node_text)
            init_feature = self.text_fusion(init_feature, text_feature)
        # <<< TEXT

        init_feature[:, 0, :] = 0  # clear the padding node

        layer_input = init_feature
        for layer in self.layers:
            hidden = F.relu(layer(layer_input, edge_list, rel_list, init_feature, relation_representations))
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        output = layer_input

        output = torch.cat([output, query.unsqueeze(1).expand(-1, output.size(1), -1)], dim=-1)

        in_batch_tensor = all_idx * torch.logical_not(mask_for_diff).int().unsqueeze(-1).expand(-1, -1, all_idx.size(-1))
        collapsed_tensor = in_batch_tensor[torch.any(in_batch_tensor != 0, dim=2)]
        feature = output.gather(1, collapsed_tensor.unsqueeze(-1).expand(-1, -1, output.size(-1)))
        score = self.mlp(feature).squeeze(-1)
        return score

    # ---- overridden copy of EntityHCNet.forward (only to route text embeds through)
    def forward(self, data, batch, relation_model):
        r_idx, entities_idx = batch[-1].T, batch[:-1].T
        edge_list, rel_list, num_nodes = data.edge_index.T, data.edge_type, data.num_nodes

        r_idx = r_idx.to(edge_list.device)
        entities_idx = entities_idx.to(edge_list.device)

        ent_text_emb = getattr(data, "ent_text_emb", None)
        rel_text_emb = getattr(data, "rel_text_emb", None)

        if self.training:
            edge_list, rel_list = self.remove_easy_edge(r_idx, entities_idx, edge_list, rel_list)
            data = tasks.build_weighted_graph(data)
            # build_weighted_graph returns a new `data` object without our
            # custom attrs (it's a plain unmodified hyper.tasks function) --
            # re-attach them so the relation_model call below still has them.
            data.ent_text_emb = ent_text_emb
            data.rel_text_emb = rel_text_emb

        if self.transductive_hcnet:
            relation_representations = None
        else:
            relation_representations = relation_model(data.meta_graph, query=r_idx[0], rel_text_emb=rel_text_emb)

        edge_list = edge_list.T
        score = self.inference(
            r_idx, entities_idx, edge_list, rel_list, num_nodes,
            relation_representations, ent_text_emb=ent_text_emb,
        )
        return score


class TextHYPER(nn.Module):
    """
    Drop-in analogue of `hyper.models.HYPER`, wired to the text-aware
    sub-modules above. Config shape matches the original `HYPER` plus a
    `text_dim` / `fusion_type` / `text_dropout` entry in each sub-model cfg
    (see config/inference/HYPER_text_inductive.yaml).
    """

    def __init__(self, rel_model_cfg, entity_model_cfg):
        super().__init__()
        self.relation_model = TextRelHCNet(**rel_model_cfg)
        self.entity_model = TextEntityHCNet(**entity_model_cfg)

    def forward(self, data, batch):
        score = self.entity_model(data, batch, self.relation_model)
        return score
