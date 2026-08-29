#!/usr/bin/env python3
"""
Script to reify a hypergraph into a knowledge graph with only binary relations.
This transforms hyper-edges of any arity into a set of binary edges using the 
reification approach.
"""

import os
import argparse
from collections import defaultdict

def read_hypergraph_file(file_path):
    """Read a hypergraph file and return a list of hyper-edges."""
    hyper_edges = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:  # Ensure there's at least a relation and one entity
                relation = parts[0]
                entities = parts[1:]
                hyper_edges.append((relation, entities))
    return hyper_edges

def reify_hypergraph(hyper_edges):
    """
    Reify a hypergraph into a set of binary relations.
    
    For each hyper-edge r(e1, e2, ..., en), create:
    1. A new entity node representing the hyper-edge: edge_id
    2. Binary relations connecting this new entity to each original entity:
       - hasEntity1(edge_id, e1)
       - hasEntity2(edge_id, e2)
       - ...
       - hasEntityN(edge_id, en)
    3. A relation type triple: hasRelationType(edge_id, r)
    """
    binary_edges = []
    edge_counter = 0
    
    for relation, entities in hyper_edges:
        # Create a unique identifier for this hyper-edge
        edge_id = f"_edge_{edge_counter}"
        edge_counter += 1
        
        # Add relation type triple
        binary_edges.append(("hasRelationType", edge_id, relation))
        
        # Add entity position triples
        for i, entity in enumerate(entities):
            position_relation = f"hasEntity{i+1}"
            binary_edges.append((position_relation, edge_id, entity))
    
    return binary_edges

def write_binary_edges(binary_edges, output_file):
    """Write the binary edges to a file."""
    with open(output_file, 'w') as f:
        for relation, subject, object in binary_edges:
            f.write(f"{relation}\t{subject}\t{object}\n")

def process_dataset(input_dir, output_dir):
    """Process all dataset files in the input directory with globally unique edge names."""
    os.makedirs(output_dir, exist_ok=True)

    # Files to process for naming (ignore hypergraph_inference.txt)
    split_files = ['train.txt', 'valid.txt', 'test.txt', 'aux.txt']
    all_hyperedges = []
    edge_name_map = dict()
    edge_counter = 0

    # 1. Collect all unique hyperedges across splits (ignore inference)
    for file_name in split_files:
        input_file = os.path.join(input_dir, file_name)
        if os.path.exists(input_file):
            hyper_edges = read_hypergraph_file(input_file)
            for he in hyper_edges:
                key = (he[0], tuple(he[1]))  # (relation, tuple(entities))
                if key not in edge_name_map:
                    edge_name_map[key] = f"_edge_{edge_counter}"
                    edge_counter += 1

    # 2. Now process each split, using the global edge_name_map
    for file_name in split_files:
        input_file = os.path.join(input_dir, file_name)
        if os.path.exists(input_file):
            output_file = os.path.join(output_dir, file_name)

            print(f"Processing {input_file}...")
            hyper_edges = read_hypergraph_file(input_file)
            binary_edges = []
            for he in hyper_edges:
                key = (he[0], tuple(he[1]))
                # Only assign edge name if it exists in the mapping (i.e., not a new edge in inference)
                edge_id = edge_name_map.get(key)
                if edge_id is None:
                    # For inference, assign a new unique edge name
                    edge_id = f"_edge_{edge_counter}"
                    edge_counter += 1
                # Reification logic (inline, to use the correct edge_id)
                # Add relation type triple
                binary_edges.append(("hasRelationType", edge_id, he[0]))
                # Add entity position triples
                for i, entity in enumerate(he[1]):
                    position_relation = f"hasEntity{i+1}"
                    binary_edges.append((position_relation, edge_id, entity))
            write_binary_edges(binary_edges, output_file)
            print(f"Created binary KG file: {output_file}")
            print(f"  Original hyper-edges: {len(hyper_edges)}")
            print(f"  Generated binary edges: {len(binary_edges)}")

def analyze_dataset(input_dir):
    """Analyze the dataset to understand its structure."""
    stats = {
        'total_edges': 0,
        'arity_distribution': defaultdict(int),
        'relation_distribution': defaultdict(int)
    }
    
    for file_name in ['train.txt', 'valid.txt', 'test.txt', 'aux.txt']:
        input_file = os.path.join(input_dir, file_name)
        if os.path.exists(input_file):
            hyper_edges = read_hypergraph_file(input_file)
            stats['total_edges'] += len(hyper_edges)
            
            for relation, entities in hyper_edges:
                arity = len(entities)
                stats['arity_distribution'][arity] += 1
                stats['relation_distribution'][relation] += 1
    
    print("\nDataset Analysis:")
    print(f"Total hyper-edges: {stats['total_edges']}")
    
    print("\nArity Distribution:")
    for arity, count in sorted(stats['arity_distribution'].items()):
        percentage = (count / stats['total_edges']) * 100
        print(f"  Arity {arity}: {count} edges ({percentage:.2f}%)")
    
    print("\nTop 10 Relations:")
    top_relations = sorted(stats['relation_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
    for relation, count in top_relations:
        percentage = (count / stats['total_edges']) * 100
        print(f"  {relation}: {count} edges ({percentage:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Reify hypergraph to binary knowledge graph')
    parser.add_argument('--input_dir', type=str, default='WD-25',
                        help='Input directory containing hypergraph files')
    parser.add_argument('--output_dir', type=str, default='WD-25-binary',
                        help='Output directory for binary knowledge graph files')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze the dataset without creating binary files')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_dataset(args.input_dir)
    else:
        analyze_dataset(args.input_dir)
        process_dataset(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
