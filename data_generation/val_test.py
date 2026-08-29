import os
import random
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--seed', default = 1, type = int)
parser.add_argument('--no_save', default=False, action='store_true')
args = parser.parse_args()
random.seed(args.seed)

full_data = os.path.join("data_generation",args.data)

print(f"PROCESSING {full_data}")

test = []
test_graph = []
test_rel = set()
test_r2ht = {}
hyper_q = {}

with open(f"{full_data}/hypergraph_inference.txt") as f:
    for line in f:
        tokens = line.strip().split()
        if not tokens:  # Skip empty lines.
            continue
        # Assume tokens[0] is the relation, and tokens[1:] are the entities.
        r = tokens[0]
        nodes = tokens[1:]
        # Store the full hyperedge as a tuple (relation, node1, node2, ...)
        hyperedge = (r, *nodes)
        test.append(hyperedge)
        test_rel.add(r)
        test_graph.append(nodes)
        
        # Build a mapping from relation to a list of node sets (i.e. hyperedges)
        if r in test_r2ht:
            test_r2ht[r].append(nodes)
        else:
            test_r2ht[r] = [nodes]
        
        # Also, build a mapping from a key (all nodes joined, ignoring relation)
        # to a list of relations connecting those nodes.
        key = tuple(nodes)  # use the full set of nodes as the key
        if key in hyper_q:
            hyper_q[key].append(r)
        else:
            hyper_q[key] = [r]

spanning_hypertree_edges = spanning_hypertree(test)

num_test = len(test)
test_msg = set()
test = set(test)  # ensure we have a set for fast removal

# Process each hyperedge in the spanning hypertree.
for edge in spanning_hypertree_edges:
    edge = tuple(edge)
    # edge is a full hyperedge: (relation, node1, node2, ..., nodeK)
    test_msg.add(edge)
    # Remove this edge's relation from test_rel (so we don't duplicate it later)
    test_rel.discard(edge[0])
    # Remove the edge from the overall test set.
    test.discard(edge)

# For any remaining relations not covered by the spanning hypertree,
# pick one representative hyperedge per relation.
for r in test_rel:
    if r in test_r2ht and test_r2ht[r]:
        edge = random.choice(test_r2ht[r])
        edge = tuple(edge)
        test_msg.add(edge)
        test.discard(edge)

# At this point, test_msg is a set of full hyperedges that:
#  - Represent a spanning connectivity structure over the hypergraph, and
#  - Ensure every relation is represented at least once.

left_test = sorted(list(test))
test_msg = sorted(list(test_msg))
random.shuffle(left_test)
remainings = int(num_test * 0.6) - len(test_msg)
test_msg += left_test[:remainings]
left_test = left_test[remainings:]

final_valid = left_test[:len(left_test)//2]
final_test = left_test[len(left_test)//2:]

if args.no_save is False:
    with open(f"{full_data}/aux.txt", "w") as f:
        for edge in test_msg:
            f.write("\t".join(map(str, edge)) + "\n")
            
    with open(f"{full_data}/valid.txt", "w") as f:
        for edge in final_valid:
            f.write("\t".join(map(str, edge)) + "\n")
            
    with open(f"{full_data}/test.txt", "w") as f:
        for edge in final_test:
            f.write("\t".join(map(str, edge)) + "\n")


# --- Function to compute statistics for a split ---
def print_stats(split_name, edges):
    unique_nodes = set()
    unique_relations = set()
    for edge in edges:
        unique_relations.add(edge[0])
        unique_nodes.update(edge[1:])
    print(f"{split_name}:")
    print(f"  Number of hyperedges: {len(edges)}")
    print(f"  Number of nodes: {len(unique_nodes)}")
    print(f"  Number of relations: {len(unique_relations)}\n")
    

# --- Print statistics for each split ---
print_stats("aux", test_msg)
print_stats("valid", final_valid)
print_stats("test", final_test)