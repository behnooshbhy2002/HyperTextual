import torch
from torch import nn
from torch.nn import functional as F
from hyper.util import static_positional_encoding, sinusoidal_positional_encoding
from . import tasks, layers





class TransductiveHCNet(nn.Module):
    def __init__(self, entity_model_cfg, num_relations):
        super(TransductiveHCNet, self).__init__()
        self.entity_model = EntityHCNet(**entity_model_cfg, transductive_hcnet=True, num_relations=num_relations)
        
    def forward(self, data, batch):

        score = self.entity_model(data, batch, relation_model=None)
        
        return score


class HYPER(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):

        super(HYPER, self).__init__()

        self.relation_model = RelHCNet(**rel_model_cfg)
        self.entity_model = EntityHCNet(**entity_model_cfg)
        

        
    def forward(self, data, batch):

        score = self.entity_model(data, batch,  self.relation_model)
        
        return score

class RelHCNet(nn.Module):
    def __init__(self, input_dim, hidden_dims,
                 short_cut=True,  num_mlp_layer=2, norm = "layer_norm", padding_idx = 0,aggregate_func = "sum", use_triton = False, **kwargs):
        super(RelHCNet,self).__init__()
        self.name = "RelHCNet"
        self.aggregate_func = aggregate_func
        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut  # whether to use residual connections between layers
        self.padding_idx = padding_idx


        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1): # num of hidden layers
            self.layers.append(layers.HypergraphLayer(self.dims[i], self.dims[i + 1], norm = norm,  aggregate_func=aggregate_func, use_triton = use_triton))
        self.feature_dim = input_dim

        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
        self.mlp = nn.Sequential(*mlp)

    def inference(self,  query_idx, edge_list, rel_list, num_nodes):

        batch_size = len(query_idx)
        
        query =  torch.ones(query_idx.shape[0], self.dims[0], device=query_idx.device, dtype=torch.float)
        index = query_idx.unsqueeze(-1).expand_as(query)
        query_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=query_idx.device)
        query_feature.scatter_add_(dim=1, 
                                   index = index.unsqueeze(1),
                                    src = query.unsqueeze(1)
                                )
        
        init_feature = query_feature

        init_feature[:, self.padding_idx, :] = 0 # clear the padding node

        # Passing in the layer:
        layer_input = init_feature
        
        for layer in self.layers:
            hidden = F.relu(layer(layer_input, edge_list, rel_list, init_feature))
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        output = layer_input

        # Remind the model which query we are looking for
        score = self.mlp(output)

        return score

    def forward(self,  relation_hypergraph, query):
        

        edge_list, rel_list, num_nodes = relation_hypergraph.edge_index, relation_hypergraph.edge_type, relation_hypergraph.num_nodes
        
        # Shift edge_list by 1 to avoid the padding node
        edge_list = torch.transpose(edge_list,0,1) + 1
        query += 1
        num_nodes +=1

        relation_feature = self.inference(query, edge_list, rel_list, num_nodes)
        query -= 1
        return relation_feature[:,1:,:]


class EntityHCNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, padding_idx=0, short_cut=True, aggregate_func="sum", concat_hidden = False, norm="layer_norm", transductive_hcnet=False, num_relations=None, use_triton = False, k_hop = False, **kwargs):

        super(EntityHCNet,self).__init__()

        self.name = "EntityHCNet"
        self.aggregate_func = aggregate_func
        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut 
        self.concat_hidden = concat_hidden
        self.padding_idx = padding_idx
        self.num_mlp_layers = 2
        self.k_hop = k_hop

        


        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1): # num of hidden layers
            self.layers.append(layers.HypergraphLayer(self.dims[i], self.dims[i + 1], norm = norm, aggregate_func=aggregate_func, use_triton = use_triton, transductive_hcnet=transductive_hcnet, num_relations=num_relations))


        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)


        self.transductive_hcnet = transductive_hcnet
        if transductive_hcnet:
            self.query = nn.Embedding(num_relations, input_dim)


    def inference(self, r_idx, entities_idx, edge_list, rel_list, num_nodes, relation_representations=None):
        r_idx = r_idx.t()
        entities_idx = entities_idx.permute(1,0,2)
        edge_list = edge_list.t()

        max_arity = len(edge_list[0])
        static_encodings = static_positional_encoding(max_arity + 1, self.input_dim)
        # Fix the encoding
        self.position = static_encodings.to(r_idx.device)
        self.position[self.padding_idx] = torch.ones(self.input_dim)

        arity = torch.ones_like(r_idx) * entities_idx.size(-1)
        
        batch_size = len(r_idx)
         # Bunch of assertion checks
        entities_idx  = torch.einsum("ijk->ikj", entities_idx)
        all_idx = entities_idx
        mask_for_diff = torch.all(all_idx[:,:,0].unsqueeze(-1).expand(-1, -1, all_idx.size(-1)) == all_idx, dim=-1)   # find for each batch, which position is searched (empty)
        
        pos_index_to_search = (mask_for_diff == False).int().argmax(dim=1)
        assert torch.all(r_idx[:, 0].unsqueeze(-1).expand(-1,r_idx.size(1)) == r_idx), "All relation types should be the same in one batch"
        assert torch.all(torch.sum(mask_for_diff, dim=-1) >= all_idx.size(1) -1 ), "Is it exactly one of the ei_idx are different?"
        assert torch.all(arity[:, 0].unsqueeze(-1).expand(-1,arity.size(1)) == arity), "All arities should be the same in one batch"
        assert torch.all(pos_index_to_search <= arity[:,0]), "The position to search should be less than arity"

        if not self.transductive_hcnet:
            query = relation_representations[torch.arange(batch_size, device=r_idx.device), r_idx[:,0]] # batch_size, hidden_dim
        else:
            assert self.query is not None, "Query should be provided for transductive HCNet"
            query = self.query(r_idx[:,0])



        init_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)
        result_tensor = torch.ones((batch_size, max_arity), dtype=torch.int, device=r_idx.device)
        range_tensor = torch.arange(max_arity, device=result_tensor.device).expand(batch_size, max_arity)
        arity_range = arity[:,0].unsqueeze(1).expand(-1, max_arity)
        
        
        mask = range_tensor < arity_range
        result_tensor *= mask
        zero_out_mask = range_tensor == pos_index_to_search.unsqueeze(1)
        result_tensor[zero_out_mask] = 0
        
        # shape: [batch_size, max_arity, 1]
        index_arity_without_self = all_idx[:,:,0] * result_tensor # Masking to find tensor
        index_arity_without_self = index_arity_without_self.unsqueeze(-1).expand(-1, -1, self.dims[0]).to(torch.int64)


        # Add relational embedding  
        query_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)

        query_feature.scatter_add_(dim=1, 
                                index=index_arity_without_self,
                                src=query.unsqueeze(1).expand(-1, max_arity, -1)
                                )
        init_feature += query_feature

        # Add positional embedding
        pos_src_index = result_tensor * torch.arange(1, max_arity+1, device=result_tensor.device).expand(batch_size, max_arity)

        pos_src = self.position[pos_src_index]
        
        # Now generate positional init feature
        pos_init_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=r_idx.device)
        # add positional encoding
        pos_init_feature.scatter_add_(dim=1,
                                index= index_arity_without_self,
                                src = pos_src
                                )
        init_feature += pos_init_feature


        init_feature[:, 0, :] = 0 # clear the padding node

        # Passing in the layer:
        layer_input = init_feature

        for layer in self.layers:
            hidden = F.relu(layer(layer_input, edge_list, rel_list, init_feature, relation_representations))
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        output = layer_input

        # Remind the model which query we are looking for
        output = torch.cat([output, query.unsqueeze(1).expand(-1, output.size(1), -1)], dim=-1)

        # output_shape is [batch_size, num_nodes, hidden_dim]
        in_batch_tensor = all_idx * torch.logical_not(mask_for_diff).int().unsqueeze(-1).expand(-1, -1, all_idx.size(-1))

        # collapsed_tensor shape is [batch_size, num_negative+1]
        collapsed_tensor = in_batch_tensor[torch.any(in_batch_tensor != 0, dim=2)]


        # feature shape is [batch_size, num_negative+1, hidden_dim]
        feature = output.gather(1, collapsed_tensor.unsqueeze(-1).expand(-1, -1, output.size(-1)))
        
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)

        return score

    def forward(self, data, batch, relation_model):
        r_idx, entities_idx = batch[-1].T, batch[:-1].T
        edge_list, rel_list, num_nodes = data.edge_index.T, data.edge_type, data.num_nodes

        r_idx = r_idx.to(edge_list.device)
        entities_idx = entities_idx.to(edge_list.device)

        
        if self.training:
            edge_list, rel_list = self.remove_easy_edge(r_idx, entities_idx, edge_list, rel_list)


            data = tasks.build_weighted_graph(data)
        if self.transductive_hcnet:
            relation_representations = None
        else:
            relation_representations = relation_model(data.meta_graph, query=r_idx[0])

        edge_list = edge_list.T

        score = self.inference(r_idx, entities_idx,  edge_list, rel_list, num_nodes, relation_representations)
        return score
    
    
    def remove_easy_edge(self, r_idx, entities_idx, edge_list, rel_list):

        # Remove the easy edges to reduce overfitting. Actually important for model to generalize
        
        # Initialize an empty mask with the same size as the edge list
        all_edge_rel = torch.cat([edge_list, rel_list.unsqueeze(-1)], dim=-1)
        easy_edge = torch.cat([entities_idx, r_idx.unsqueeze(-1)], dim=-1).flatten(0,1)
        all_edge_rel, easy_edge = all_edge_rel.transpose(0,1), easy_edge.transpose(0,1)
        index = tasks.edge_match(all_edge_rel, easy_edge)[0]
        remove_mask = ~self.index_to_mask(index, len(edge_list))
        
        # Filter out the edges that are to be removed
        filtered_edge_list = edge_list[remove_mask,:]
        filtered_rel_list = rel_list[remove_mask]
        return filtered_edge_list, filtered_rel_list
        
    def index_to_mask(self, index, size):
        index = index.view(-1)
        size = int(index.max()) + 1 if size is None else size
        mask = index.new_zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

    def extract_k_hop_subgraph(self, edge_list, rel_list, entities_idx, num_entities, k=3, padding_idx=-1):
        # Flatten and get unique seed entities from the batch's entities_idx
        seeds = entities_idx.flatten().unique(sorted=False)
        
        # Determine the number of entities, assuming self.num_entities is available
        device = edge_list.device
        
        # Initialize current_entities with seeds
        current_entities = torch.zeros(num_entities, dtype=torch.bool, device=device)
        current_entities[seeds] = True
        
        for _ in range(k):
            # Mask for valid (non-padding) entries in edge_list
            valid_entries = edge_list != padding_idx
            
            # Check if any entity in each edge is in current_entities (ignoring padding)
            current_in_edge = current_entities[edge_list] & valid_entries
            edge_mask = current_in_edge.any(dim=1)
            
            # Get all entities from the connected edges
            connected_edges = edge_list[edge_mask]
            if connected_edges.numel() == 0:
                break  # No new entities to add
            
            new_entities = connected_edges.flatten()
            new_entities = new_entities[new_entities != padding_idx].unique()
            
            # Update current_entities with new_entities
            current_entities[new_entities] = True
        
        # Determine which edges to keep: all valid entities in the edge must be in current_entities
        valid_entries = edge_list != padding_idx
        all_valid_in_current = (current_entities[edge_list] | ~valid_entries).all(dim=1)
        
        subgraph_edge_list = edge_list[all_valid_in_current]
        subgraph_rel_list = rel_list[all_valid_in_current]
        
        return subgraph_edge_list, subgraph_rel_list
