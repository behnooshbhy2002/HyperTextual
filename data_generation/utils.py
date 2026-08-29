import networkx as nx
import random
import numpy as np
import tqdm

# Removes duplicate elements from a list while preserving order
def remove_duplicate(x):
    return list(dict.fromkeys(x))

# Reads a file containing tuples and processes them into hyperedges, entities, and relations
def read_tuple(file_path, is_JF):
    print(f"Reading {file_path}")
    with open(file_path, "r") as f:
        lines = f.readlines()
    hyperedges = []  # Stores the processed hyperedges
    entity = []  # Stores all entities in the hyperedges
    relation = []  # Stores all relations in the hyperedges
    for i, line in enumerate(lines):
        tokens = line.strip().split("\t")  # Split the line by tabs
        if is_JF:
            tokens = tokens[1:]  # Ignore the first token if the dataset is JF
        edge = tuple(t for t in tokens)  # Convert tokens to a tuple
        entity.extend(edge[1:])  # Add all entities in the edge to the entity list
        relation.append(edge[0])  # Add the relation to the relation list
        hyperedges.append(edge)  # Add the edge to the hyperedges list
    return remove_duplicate(entity), remove_duplicate(relation), remove_duplicate(hyperedges)

# Gathers entities and relations from a list of hyperedges
def gather(x):
    ent = []  # Stores all entities in the hyperedges
    rel = []  # Stores all relations in the hyperedges
    for edge in x:
        # The first element is the relation.
        rel.append(edge[0])
        # The rest of the elements are entities.
        ent.extend(edge[1:])
    return remove_duplicate(ent), remove_duplicate(rel)

# Checks if two lists have no overlapping elements
def check_no_overlap(x, y):
    assert len(set(x).intersection(set(y))) == 0
    print("Done: Check no overlap")

# Writes a list of hyperedges to a file
def write(path, hyperedges):
    with open(path, 'w') as f:
        for edge in hyperedges:
            f.write("\t".join(map(str, edge)) + "\n")

# Gathers neighboring entities of a given entity within a list of hyperedges
def gather_neighbor(hyperedges, x, thr):
    neighbors = []  # Stores the neighboring entities
    for edge in hyperedges:
        # edge[0] is the relation, edge[1:] are the entities.
        if x in edge[1:]:
            # Add all entities in this hyperedge except x.
            for ent in edge[1:]:
                if ent != x:
                    neighbors.append(ent)
    neighbors = remove_duplicate(neighbors)
    if len(neighbors) > thr:
        neighbors = random.sample(neighbors, thr)
    return neighbors

# Samples entities within two hops of a given list of entities
def sample_2hop(tuples, x, thr):
    sample = set()  # Stores the sampled entities
    for e in x:
        neighbor = set([e])
        neighbor_1hop = gather_neighbor(tuples, e, thr)
        neighbor = neighbor.union(set(neighbor_1hop))

        for e1 in neighbor_1hop:
            neighbor_2hop = gather_neighbor(tuples, e1, thr)
            neighbor = neighbor.union(set(neighbor_2hop))

        sample = sample.union(neighbor)
    return sample

# Merges two lists based on a given proportion
def merge(x, y, p):
    if p >= 1:
        return y
    elif p <= 0:
        return x
    else:
        num_tot = min(len(x) / (1 - p), len(y) / p)
        random.shuffle(x)
        random.shuffle(y)
        return x[:int(num_tot * (1 - p))] + y[:int(num_tot * p)]

# Finds the largest connected component in a hypergraph
def gcc(hyperedges):
    hyperedges = [hyperedge[1:] for hyperedge in hyperedges]
    # hyperedges is a tuple of hyperedges which ignore the relation types,
    # where each hyperedge is represented as a tuple of nodes.
    
    # Create a bipartite graph representation of the hypergraph.
    B = nx.Graph()
    
    # For each hyperedge, create a node (with a unique id)
    # and connect it to every vertex in that hyperedge.
    for i, hedge in enumerate(hyperedges):
        hedge_node = f"he_{i}"  # hyperedge identifier
        B.add_node(hedge_node, bipartite=1)
        for v in hedge:
            B.add_node(v, bipartite=0)
            B.add_edge(v, hedge_node)
    
    # Find the largest connected component in the bipartite graph.
    largest_cc = max(nx.connected_components(B), key=len)
    
    # Since the bipartite graph contains both vertices and hyperedge nodes,
    # filter out the hyperedge nodes (here, nodes starting with "he_")
    vertices_cc = {n for n in largest_cc if not str(n).startswith("he_")}
    
    return vertices_cc

# A class for union-find operations
class UnionFind:
    def __init__(self, elements):
        self.parent = {x: x for x in elements}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[ry] = rx

# Computes a spanning hypertree from a collection of hyperedges
def spanning_hypertree(hyperedges):
    """
    Computes a spanning hypertree (a set of hyperedges connecting all vertices)
    from a collection of hyperedges.
    
    Parameters:
      hyperedges: iterable of tuples, where each tuple is of the form
                  (relation, node1, node2, ..., node_k)
    
    Returns:
      A list of hyperedges that (greedily) connect all vertices.
    """
    # First, collect all vertices present in the hypergraph.
    vertices = set()
    for edge in hyperedges:
        # We assume the first element is the relation; the rest are vertices.
        vertices.update(edge[1:])
    vertices = list(vertices)
    
    # Initialize union-find on all vertices.
    uf = UnionFind(vertices)
    
    # Prepare a list to hold the spanning hyperedges.
    spanning = []
    
    # Shuffle the hyperedges to introduce randomness.
    hyperedges_shuffled = list(hyperedges)
    random.shuffle(hyperedges_shuffled)
    
    # Greedily add hyperedges that connect previously disconnected vertices.
    for edge in hyperedges_shuffled:
        nodes = edge[1:]
        if len(nodes) < 2:
            # A hyperedge with fewer than 2 vertices cannot connect vertices.
            continue
        
        # Check if this hyperedge would connect two different components.
        # We take the first vertex as a reference.
        rep = uf.find(nodes[0])
        # If at least one other vertex is in a different component, then adding
        # this hyperedge will connect parts of the hypergraph.
        if any(uf.find(v) != rep for v in nodes[1:]):
            spanning.append(edge)
            # Union all nodes in this hyperedge.
            for v in nodes[1:]:
                uf.union(nodes[0], v)
        
        # If all vertices are now connected, we have a spanning hypertree.
        if len({uf.find(v) for v in vertices}) == 1:
            break
            
    return spanning

def remove_binary_relations(hyperedges, percentage):
    # Identify all the hyperedges with binary relations.
    # Identify binary relations - those with exactly 3 elements (1 relation + 2 entities)
    binary_hyperedges = [edge for edge in hyperedges if len(edge) == 3]
    num_remove = int(len(binary_hyperedges) * percentage)
    
    if num_remove > 0:
        # Randomly select binary relations to remove
        indices_to_remove = np.random.choice(len(binary_hyperedges), num_remove, replace=False)
        remove_set = set(binary_hyperedges[i] for i in indices_to_remove)
        
        # Filter out the removed relations
        result = [edge for edge in hyperedges if edge not in remove_set]
    else:
        result = hyperedges.copy()
    
    print(f"Removed {num_remove} edges with binary relations.")

    return result
