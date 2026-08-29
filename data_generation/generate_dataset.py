### Generate new knowledge graph ###
from utils import *
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_src', type=str)
parser.add_argument('--data_tgt', type=str)
parser.add_argument('--n_train', type=int)
parser.add_argument('--n_test', type=int)
parser.add_argument('--p_rel', type=float)
parser.add_argument('--p_tri', type=float)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--no_save', default=False, action='store_true')
parser.add_argument('--percentage_binary_remove', type=float, default=0.0)
args = parser.parse_args()

seed = int(100 * args.n_train * args.n_test / args.p_rel * args.seed)

random.seed(seed)

### Read entities/relations ###
tuples_all = []
for split in ["train", "test", "valid"]:
    _, _, tuples = read_tuple(f"hypergraph_dataset/{args.data_src}/{split}.txt", is_JF=(args.data_src == "JF17K" and split == "test"))
    tuples_all += tuples

### Remove binary relations ###
if args.percentage_binary_remove > 0:
    tuples_all = remove_binary_relations(tuples_all, args.percentage_binary_remove)

### Take GCC ###
gcc_all = gcc(tuples_all)
entity, relation, hyperedges = [], [], []
for edge in tuples_all:
    # Each edge is expected to be of the form: (relation, entity1, entity2, ...)
    rel = edge[0]
    ents = edge[1:]
    # Use the first entity as the 'head' for the gcc condition
    if ents and ents[0] in gcc_all:
        entity.extend(ents)
        relation.append(rel)
        hyperedges.append(edge)
        

entity = remove_duplicate(entity)
relation = remove_duplicate(relation)

### Split relation set into train/valid/test ###
num_relation = len(relation)
random.shuffle(relation)
relation_test = relation[:int(num_relation * args.p_rel)]
relation_train = relation[int(num_relation * args.p_rel):]

relation_test = set(relation_test)
relation_train = set(relation_train)

### Sample neighbors from train seeds ###
seed_train = random.sample(entity, args.n_train)
entity_train = sample_2hop(tuples, seed_train, 50)

### Generate train set ###
train_all = []
for edge in tuples:
    rel = edge[0]
    ents = edge[1:]
    if rel in relation_train and all(e in entity_train for e in ents):
        train_all.append(edge)

### Take GCC ###
gcc_train = gcc(train_all)
train = []
for edge in train_all:
    # Assuming the hyperedge is structured as (relation, head, ...),
    # we use the first entity (at index 1) as the "head" for the condition.
    if edge[1] in gcc_train:
        train.append(edge)
random.shuffle(train)

### Remove train entities ###
tuples_p = []
for edge in tuples:
    # edge is structured as (relation, entity1, entity2, ...)
    ents = edge[1:]
    # Check if none of the entities are in gcc_train.
    if not any(e in gcc_train for e in ents):
        tuples_p.append(edge)

entity_p, relation_p = gather(tuples_p)

### Sample neighbors from valid seeds ###
seed_test = random.sample(entity_p, args.n_test)
entity_test = sample_2hop(tuples_p, seed_test, 50)

### Generate test set ###
test_x = []
test_y = []
for edge in tuples_p:
    # Each hyperedge is structured as (relation, entity1, entity2, ...)
    rel = edge[0]
    ents = edge[1:]
    # Only consider edges where all entities are in the test set.
    if all(e in entity_test for e in ents):
        if rel in relation_train:
            test_x.append(edge)
        elif rel in relation_test:
            test_y.append(edge)

### Merge X_test and Y_test ###
test_all = merge(test_x, test_y, args.p_tri)

### Take GCC ###
gcc_test = gcc(test_all)  # Compute the giant connected component on the test hypergraph
test = []
for edge in test_all:
    # Each edge is structured as (relation, entity1, entity2, ...)
    ents = edge[1:]
    # If at least one entity is in the GCC, we include the hyperedge.
    if any(e in gcc_test for e in ents):
        test.append(edge)

random.shuffle(test)

### Check no overlap ###
check_no_overlap(gcc_train, gcc_test)

# For the training hypergraph (train)
train_nodes = set()
train_relations = set()
max_arity_train = 0  # maximum number of entity arguments in a training edge
for edge in train:
    train_relations.add(edge[0])
    # Update nodes with all entities in the edge (skip the relation at index 0)
    train_nodes.update(edge[1:])
    # Compute the arity (number of entities in the edge)
    current_arity = len(edge) - 1
    if current_arity > max_arity_train:
        max_arity_train = current_arity


# For the inference/test hypergraph (test)
test_nodes = set()
test_relations = set()
max_arity_test = 0  # maximum number of entity arguments in a test edge
for edge in test:
    test_relations.add(edge[0])
    test_nodes.update(edge[1:])
    current_arity = len(edge) - 1
    if current_arity > max_arity_test:
        max_arity_test = current_arity



print("Number of hyperedges in Train hyperGraph:", len(train))
print("Number of nodes in Train hyperGraph:", len(train_nodes))
print("Number of relations in Train hyperGraph:", len(train_relations))
print("Max arity:", max_arity_train)
print("---"*30)
print("Number of hyperedges in Inference hyperGraph:", len(test))
print("Number of nodes in Inference hyperGraph:", len(test_nodes))
print("Number of relations in Inference hyperGraph:", len(test_relations))
print("Max arity:", max_arity_test)
print("---"*30)

# Calculate relation overlap statistics
common_relations = train_relations.intersection(test_relations)
only_in_train = train_relations - test_relations
only_in_test = test_relations - train_relations

overlap_percentage = 0
if test_relations:
    overlap_percentage = (len(common_relations) / len(test_relations)) * 100

print("Relation Overlap Analysis:")
print(f"Common relations: {len(common_relations)} ({overlap_percentage:.2f}%)")
print(f"Relations only in train: {len(only_in_train)}")
print(f"Relations only in inference: {len(only_in_test)}")

### Save files ###
if not args.no_save:
	save_dir = f"data_generation/{args.data_tgt}/"
	os.makedirs(save_dir, exist_ok=True)
	write(save_dir + 'train.txt', train)
	write(save_dir + 'hypergraph_inference.txt', test)
