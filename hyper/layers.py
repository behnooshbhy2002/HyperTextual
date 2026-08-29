import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.utils import scatter

from hyper.util import static_positional_encoding, preprocess_triton_hypergraph, sinusoidal_positional_encoding
from hyper.tasks import forward_mapping


# Turn on triton if possible
class HypergraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, aggregate_func = "sum", norm = "layer_norm",  use_triton = False, transductive_hcnet = False, num_relations = None):
        super(HypergraphLayer, self).__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.norm_type = norm
        self.use_triton = use_triton
        self.aggregate_func = aggregate_func 

        if norm == "layer_norm":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        self.relational_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )


        self.transductive_hcnet = transductive_hcnet
        if transductive_hcnet:
            assert num_relations is not None, "Number of relations must be provided for transductive HCNet"
            self.rel_embedding = nn.Embedding(num_relations, in_channels)
            

        

    def forward(self, node_features, edge_list, rel, init_feature, relation_representations=None):
        max_arity = len(edge_list[0])
        device = node_features.device
        # positional encoding is calculated at forward pass
        static_encodings = static_positional_encoding(max_arity, self.in_channels).to(device)

        self.pos_embedding = static_encodings
        self.pos_embedding[0] = torch.ones(self.in_channels, device = device) # set the padding to 1

        # now compute relational embeddings
        if relation_representations is not None:
            # Entity encoder
            relation_representations = self.relational_mlp(relation_representations)
            
        else:
            if not self.transductive_hcnet:

                # relation_encoder
                max_relation = int(torch.sqrt(torch.max(rel) + 1))
                
                positional_encoding = sinusoidal_positional_encoding(max_relation, self.in_channels//2).to(device)
                
                rel_1 = torch.arange(max_relation, device = device).repeat(max_relation).to(device)
                rel_2 = torch.arange(max_relation, device = device).repeat_interleave(max_relation).to(device)
                concat_rel_embed = torch.cat([positional_encoding[rel_1],positional_encoding[rel_2]], dim = 1)
                # concat_rel_embed is a tensor of shape [max_relation ** 2, in_channels], where the k-th index is f(x,y) = k
            

                # Passing through a MLP
                transformed_rel_embed = self.relational_mlp(concat_rel_embed)

                # sort the MLP
                index = [i for i, _ in sorted(enumerate([forward_mapping(i.item(),j.item()) for i,j in zip(rel_1,rel_2)]), key=lambda x: x[1])]
                transformed_rel_embed = transformed_rel_embed[index]
                self.rel_embedding = transformed_rel_embed

            else:
                assert self.rel_embedding is not None, "Relation embeddings must be provided for transductive HCNet"


        batch_size, node_size, _ = node_features.shape
        
             
        mask = torch.ones_like(node_features)
        mask[:, 0, :] = 0
        node_features = node_features * mask
        
        if self.use_triton:
            from .rspmm.triton_rspmm import HyperRelConvSumAggr
            pos_embedding = self.pos_embedding.unsqueeze(1).expand(-1, batch_size, -1).flatten(1) # expand the positional encoding for batch, and compress with the feature size
            if relation_representations is None:
                if self.transductive_hcnet:
                    relation_vector = self.rel_embedding.weight.unsqueeze(1).expand(-1, batch_size, -1).flatten(1).transpose(0,1)
                else:
                    relation_vector = self.rel_embedding.unsqueeze(1).expand(-1, batch_size, -1).flatten(1).transpose(0,1) # expand the relation embedding for batch, and compress with the feature size
            else:
                relation_vector = relation_representations.permute(1, 0, 2).reshape(relation_representations.shape[1], -1)
            edge_list_trans = edge_list.transpose(0,1)
            node_features_flatten = node_features.transpose(0,1).flatten(1)
            rowptr, indices, etypes, pos_index, _ = preprocess_triton_hypergraph(edge_list_trans, rel, num_node = node_size)
            if self.aggregate_func == "sum":
                out  = HyperRelConvSumAggr.apply(node_features_flatten, rowptr, indices, node_size, etypes, relation_vector, pos_embedding, pos_index, 0)
            else:
                raise ValueError("For now, the Triton kernel only supports sum aggr. Unknown aggregation function `%s`" % self.aggregate_func)
            out = out.view(node_size, batch_size, -1).transpose(0,1)
        else:
            message = self.messages(node_features, relation_representations, edge_list, rel)
            out = self.aggregates(message, edge_list, node_features)
            out[:, 0, :] = 0 # Clear the padding node for learning
        out = (self.linear(torch.cat([out, init_feature], dim=-1)))
    
        if self.norm_type == "layer_norm":
            out = self.norm(out)

        return out

    def messages(self, node_features, relation_vector, hyperedges, relations):
        device = node_features.device
        # Set the node feature of node 0 to be always 0 so that it does not contribute to the messages

        batch_size, _, input_dim = node_features.shape
        edge_size, max_arity = hyperedges.shape

        # Create a batch index array
        batch_indices = torch.arange(batch_size, device=hyperedges.device)[:, None, None]  # Shape: [batch_size, 1, 1]

        # Repeat batch indices to match the shape of hyperedges
        # New shape of batch_indices: [batch_size, edge_size, max_arity]
        batch_indices = batch_indices.repeat(1, hyperedges.shape[0], hyperedges.shape[1]) # TODO: maybe replace with torch.expand

        # Use advanced indexing to gather node features
        # The resulting shape will be [batch_size, edge_size, max_arity, input_dim]
        sum_node_positional = node_features[batch_indices, hyperedges]


        # Compute positional encodings for nodes in each hyperedge
        # [batch_size, edge_size, max_arity, input_dim]
        positional_encodings = self.computer_pos_encoding(hyperedges, batch_size, device)

        # Sum node features and positional encodings
        # Final shape: [batch_size, edge_size, max_arity, input_dim]
        sum_node_positional = sum_node_positional + positional_encodings

        # sum_node_positional is actually the ej+pj for each node that is located in each edge, indicated by its max_arity
        # we need to produce another [batch_size, edge_size, max_arity, input_dim], that compute *_{j \neq i}(e_j+p_j), which replace the i pos
        # We can do this by a clever "shift" operation. Compute the cumulative product in both directions [batch_size, edge_size, max_arity, input_dim]
        messages = self.all_but_one_trick(sum_node_positional, batch_size, edge_size, input_dim, device)
        
        # Get relation vectors for each edge and expand
        # Shape: [edge_size] -> [batch_size,  edge_size, max_arity, input_dim]
        if relation_vector is None:
            if self.transductive_hcnet:
                relation_vectors = self.rel_embedding(relations).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, max_arity, -1)
            else:
                relation_vectors = self.rel_embedding[relations].unsqueeze(0).unsqueeze(2).expand(batch_size, -1, max_arity, -1)
        else:
            relation_vectors = relation_vector.index_select(1, relations)
            relation_vectors = relation_vectors.unsqueeze(2).expand(-1, -1, max_arity, -1)

        messages = messages * relation_vectors

        # shape: [batch_size,  edge_size, max_arity, input_dim]
        return messages

    def aggregates(self, messages, hyperedges, node_features):
        # Messages has shape [batch_size,  edge_size, max_arity, input_dim], where each edges stores max_arity messages, each belongs to the position of the node at that max_arity
        # hyperedges has shape [batch_size, edge_size, max_arity], where each edge stores the node index that belongs to that edge
        # relations has shape [batch_size, edge_size], where each edge stores the relation index that belongs to that edge
        # node_features has shape [batch_size, node_size, input_dim], where each node stores the feature vector of that node
        batch_size, node_size, input_dim = node_features.shape
        edge_size, max_arity = hyperedges.shape

        # Expand and reshape messages for gathering
        # Shape: [batch_size, edge_size, max_arity, input_dim] -> [batch_size, edge_size * max_arity, input_dim]
        messages_expanded = messages.view(batch_size, edge_size * max_arity, input_dim)

        # Gather messages based on hyperedges indices
        # New shape after gather: [batch_size, node_size, input_dim]

        node_aggregate = scatter(messages_expanded, hyperedges.flatten(), dim = 1, reduce = "sum", dim_size=node_size)
        
        # The output is a tensor of shape [batch_size, node_size, input_dim], where each node stores the aggregated message from all the edges that it belongs to
        return node_aggregate


        
    def all_but_one_trick(self, sum_node_positional, batch_size, edge_size, input_dim, device):
        cumprod_forward = torch.cumprod(sum_node_positional, dim=2)
        cumprod_backward = torch.cumprod(sum_node_positional.flip(dims=[2]), dim=2).flip(dims=[2])

        # Shift and combine
        shifted_forward = torch.cat([torch.ones(batch_size, edge_size, 1, input_dim).to(device), cumprod_forward[:, :, :-1, :]], dim=2)
        shifted_backward = torch.cat([cumprod_backward[:, :, 1:, :], torch.ones(batch_size, edge_size, 1, input_dim).to(device)], dim=2)

        # Combine the two shifted products
        return shifted_forward * shifted_backward

    def computer_pos_encoding(self, hyperedges, batch_size, device):
        
        sequence_tensor = torch.arange(1, hyperedges.size(1) + 1, device = device).unsqueeze(0)
        # Apply the sequence tensor to the non-zero elements
        pos_node_in_edge = torch.where(hyperedges != 0, sequence_tensor, torch.zeros_like(hyperedges, device = device))

        # [batch_size, edge_size, max_arity, input_dim]
        return self.pos_embedding[pos_node_in_edge].unsqueeze(0).expand(batch_size, -1, -1, -1)
    

