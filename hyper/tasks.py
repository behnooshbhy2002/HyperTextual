from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch

# defining the mapping function for two edge type to one edge type
def forward_mapping(i,j):
    if i == j: return (i+1)**2 - 1
    if i > j: return (i+1)**2 - 1 - 2*(i-j)
    if i < j: return (j+1)**2 - 2*(j-i)


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    # assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    # Number of nodes in each hyperedge (arity of hyperedges)
    num_nodes_in_edge = batch.size(1) - 1  
    pos_indices = batch.t()[:-1]  # Shape: (k, batch_size)
    r_index = batch.t()[-1]  # Shape: (batch_size)
    # Mask indicating non-padding positions (1 if not 0, 0 if 0)
    non_zero_mask = (pos_indices != 0)  # Shape: (num_nodes_in_edge, batch_size)


    # strict negative sampling vs random negative sampling
    if strict:
        # Generate masks for each node position in the hyperedge
        masks = strict_negative_mask(data, batch)  # Returns a list of masks

        neg_indices = []
        for i, mask in enumerate(masks):
            # Adjust the mask for the current node position
            # Only sample for non-zero positions
            if not non_zero_mask[i].any():  # Skip positions that are all zeros
                neg_indices.append(None)
                continue

            current_mask = mask & non_zero_mask[i].unsqueeze(-1)  

            # Extract valid candidates
            neg_candidate = current_mask.nonzero()[:, 1]
            num_candidate = current_mask.sum(dim=-1)


            valid_mask = num_candidate > 0  # Shape: [8]

            # Expand the mask to match the shape of the indexed tensor
            valid_mask = valid_mask.unsqueeze(-1).expand(-1, num_negative)  # Shape: [8, num_negative]

            # Step 1: Generate random numbers
            rand = torch.rand(len(current_mask), num_negative, device=batch.device)  # Shape: [8, num_negative]

            # Step 2: Scale random numbers by num_candidate only for valid rows
            scaled_rand = (rand * num_candidate.unsqueeze(-1)).long()  # Shape: [8, num_negative]

            # Step 3: Compute offsets only for valid rows
            offsets = (num_candidate.cumsum(0) - num_candidate).unsqueeze(-1)  # Shape: [8, 1]

            # Add offsets
            index = scaled_rand + offsets

            # Step 4: Mask out invalid rows
            index[~valid_mask] = 0  # Rows with no candidates are set to 0 (or any other placeholder)

            
            # Map sampled indices back to candidate set
            neg_index = neg_candidate[index]
            neg_indices.append(neg_index)
    else:
        # Random negative sampling
        neg_indices = []
        for _ in range(num_nodes_in_edge):
            if not non_zero_mask[i].any():  # Skip positions that are all zeros
                neg_indices.append(None)
                continue
            neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
            neg_indices.append(neg_index)

    # Prepare positive indices for all nodes
    pos_index_list = []
    # Replace the positive indices with negative samples for each node position
    for i in range(len([item for item in neg_indices if item is not None])):
        # ablate on i-th node
        pos_index = pos_indices.unsqueeze(-1).repeat(1,1,neg_indices[i].shape[1]+1)  # Shape: ( max_arity, batch_size,num_negative+1)
            # Replace positive indices at the i-th position with negative samples

        # generate a mask of all true with shape of pos_index
        mask = torch.ones_like(pos_index, dtype=torch.bool)

        for val in range(batch_size):
            if neg_indices[i][val][0] == 0:
                mask[i,val] = False
                
                continue
            # Clone pos_indices to avoid modifying the original
            pos_index[i,val,1:] = neg_indices[i][val] # shape of neg_indices[i] is (batch_size, num_negative)
        
        # propagate_mask
        mask = torch.all(mask[:,:,0],dim = 0)
        pos_index = pos_index[:,mask,:]
        
        modified_r_index = r_index[mask].unsqueeze(0).repeat(pos_index.shape[-1],1).t().unsqueeze(0)
        stack_temp=torch.cat([pos_index, modified_r_index], dim=0)

        pos_index_list.append(stack_temp)
    negative_samples = torch.concat(pos_index_list, dim=1)  # (max_arity, new_batch_size, num_negative+1)
    return negative_samples # (max_arity+1, new_batch_size, num_negative+1)



def all_negative(data, batch):
    """
    Generate all negative samples for a hypergraph by replacing each node in the hyperedge
    with all possible nodes in the graph.

    Args:
        data: The data object containing the hypergraph information.
            - data.num_nodes: Total number of nodes in the hypergraph.
        batch: Tensor of shape (batch_size, k), where each row represents a hyperedge in the batch.
            - batch.device: The device on which tensors are allocated.
        
    Returns:
        negative_samples: A list of tensors, one for each node position.
            Each tensor is of shape (batch_size, data.num_nodes, k), containing negative samples
            generated by replacing the node at that position with all possible nodes.
    """
    batch_size, _ = batch.size()[0], batch.size()[1]-1
    pos_indices = batch  # Shape: (batch_size, k)
    negative_samples = []
    all_nodes = torch.arange(1, data.num_nodes, device=batch.device)
    max_arity_here = torch.sum(torch.any(pos_indices.T[:-1] != 0, dim=1))
    for i in range(max_arity_here):
        # Expand the positive indices to match the shape needed for replacement
        pos_indices_expanded = pos_indices.unsqueeze(1).expand(-1, data.num_nodes-1, -1)  # Shape: (batch_size, num_nodes, k)
        temp = []
        for val in range(batch_size):
            if pos_indices[val,i] > 0:  # Check to avoid any invalid indexing when index is 0
                neg_indices = pos_indices_expanded[val].clone() 
                neg_indices[:,i] = all_nodes 
                temp.append(neg_indices)
        
        neg_indices = torch.stack(temp,dim=0).permute(2,0,1)
        negative_samples.append(neg_indices)
    return negative_samples




def strict_negative_mask(data, batch):
    # Number of nodes in each hyperedge (arity of hyperedges)
    num_nodes_in_edge = batch.size(1) - 1  
    pos_indices = batch.t()[:-1]  # Shape: (k, batch_size)
    r_index = batch.t()[-1]  # Shape: (batch_size)
    masks = []
    for i in range(num_nodes_in_edge):
        edge_other_indices = data.edge_index[torch.arange(num_nodes_in_edge) != i]
        edge_index = torch.cat([edge_other_indices, data.edge_type.unsqueeze(0)],dim=0)
        query_index = pos_indices[torch.arange(num_nodes_in_edge) != i]  # Shape: (k-1, batch_size)
        query_index = torch.cat([query_index, r_index.unsqueeze(0)], dim=0)  # Shape: (k, batch_size)
        edge_id, num_i_truth = edge_match(edge_index, query_index)
        i_truth_index = data.edge_index[i, edge_id]
        sample_id = torch.arange(len(num_i_truth), device=batch.device).repeat_interleave(num_i_truth)
        mask = torch.ones(len(num_i_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
        mask[sample_id, i_truth_index] = 0
        mask.scatter_(1, pos_indices[i].unsqueeze(-1), 0)
        masks.append(mask)
    return masks




def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1) - 1) # -1 because we always start from 1 entitiy
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking





def generate_subarity_matrix(arity, edge_index, edge_type, num_nodes, num_rels, max_arity, device):
    assert 0 <= arity < max_arity, f"Invalid arity 0 <= {arity} < {max_arity}"
    Eh = torch.vstack([edge_index[arity], edge_type]).T.unique(dim=0)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])
    assert not (Dh[Eh[:, 0]] == 0).any()
    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    return Eh, EhT, Dh


def build_weighted_graph(graph):

    # we build the weighted graph with inverse edges 

    edge_index, edge_type = graph.edge_index, graph.edge_type # (max_arity, num_edge), (num_edge)
    num_nodes, num_rels, max_arity = graph.num_nodes, graph.num_relations, graph.max_arity
    device = graph.device

    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    subarity_matrices = []
    for arity in range(max_arity):
        Eh, EhT, Dh = generate_subarity_matrix(arity, edge_index, edge_type, num_nodes, num_rels, max_arity, device)
        subarity_matrices.append([EhT, Eh, Dh])

    # Carry out sparse matrix multiplication
    resulting_edges = []
    for i, (EhT_i, _, _) in enumerate(subarity_matrices):
        for j, (_, Eh_j,_) in enumerate(subarity_matrices):
            A = torch.sparse.mm(EhT_i, Eh_j).coalesce()
            edges = torch.cat([A.indices().T, 
                               torch.zeros(A.indices().T.shape[0], 1, dtype=torch.long).fill_(forward_mapping(i,j)).to(device),  # position for the first arity
                               ],dim=1)  
            resulting_edges.append(edges)

    # define the meta_graph
    meta_graph = Data(
        edge_index=torch.cat([edges[:, [0, 1]].T for edges in resulting_edges], dim=1), 
        edge_type=torch.cat([edges[:, 2] for edges in resulting_edges], dim=0),
        num_nodes=num_rels, 
        num_relations=max_arity**2
    )
    graph.meta_graph = meta_graph
    return graph

  


