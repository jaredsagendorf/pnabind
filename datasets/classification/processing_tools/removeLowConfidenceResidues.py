#!/usr/bin/env python

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("data_files")
arg_parser.add_argument("source_directory")
arg_parser.add_argument("target_directory")
arg_parser.add_argument("--suffix", default="")
arg_parser.add_argument("--save_separate_components", action="store_true",
        help="write disconnected regions as separate PDB files")
arg_parser.add_argument("--confidence_threshold", default=65.0, type=float,
        help="residues below this threshold will be flagged for removal")
arg_parser.add_argument("--component_size_threshold", default=50, type=int,
        help="minimum size of a connected region to be kept")
arg_parser.add_argument("--low_confidence_length_threshold", default=6, type=int,
        help="consecutive low confidence regions longer than this will be removed")
arg_parser.add_argument("--show_progress", action="store_true")
arg_parser.add_argument("-o", "--output", default="structure_stats.csv")
ARGS = arg_parser.parse_args()

from tqdm import tqdm
import os
import numpy as np
import networkx as nx
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBIO import Select
from Bio.PDB import NeighborSearch

class SelectComponent(Select):
    def __init__(self, component):
        self.residues = set(component)
    
    def accept_residue(self, residue):
        return residue.get_id() in self.residues

def makeGraph(structure, distance_cutoff=3.5, level='R'):
    alist = list(structure.get_atoms())
    kdt = NeighborSearch(alist)
    G = nx.Graph()
    
    # add graph nodes
    for residue in structure.get_residues():
        G.add_node(residue.get_id())
    
    # add graph edges
    pairs = kdt.search_all(distance_cutoff, level=level)
    for pair in pairs:
        G.add_edge(pair[0].get_id(), pair[1].get_id())
    
    return G

parser = PDBParser()
io = PDBIO()

# write out structure info
OUT = open(ARGS.output, "w")
OUT.write("{},{},{},{},{}".format("structure_id", "low_confidence_ratio", "num_components", "keep_ratio", "full_chain_length"))

# loop over PDB files in ARGS.source_directory
file_list = open(ARGS.data_files)
if ARGS.show_progress:
    iterator = tqdm(file_list.readlines())
else:
    iterator = file_list

for f in iterator:
    f = f.strip()
    
    # get file names
    structure_id = ".".join(f.split('.')[:-1]) # strip file extention
    fname = "{}{}.pdb".format(structure_id, ARGS.suffix)
    in_file = os.path.join(ARGS.source_directory, f)
    out_file = os.path.join(ARGS.target_directory, fname)
    
    if os.path.exists(out_file):
        # output file already exists
        continue
    
    # load predicted structure
    structure = parser.get_structure(structure_id, in_file)
    chain = structure[0]['A']
    num_residues = len(chain)
    
    # get residue scores
    scores = []
    low_residues = []
    residues = chain.get_list()
    for i in range(len(residues)):
        score = residues[i].child_list[0].bfactor
        if score < ARGS.confidence_threshold:
            low_residues.append(i)
        scores.append(score)
    scores = np.array(scores)
    
    # get consecutive stretches of low residues
    count = 1
    keep = set()
    for j in range(1, len(low_residues)):
        if low_residues[j]-low_residues[j-1] == 1:
            count += 1
        else:
            if count <= ARGS.low_confidence_length_threshold:
                # store consecutive residues
                keep.update(range(low_residues[j-count], low_residues[j-1]+1))
            count = 1
    
    # remove residues outside stretches
    for i in low_residues:
        if i in keep:
            continue
        chain.detach_child(residues[i].get_id())
    
    num_components = 0
    num_kept_residues = 0
    kept_components = []
    if len(chain) > 0:
        # get connected components of structure
        G = makeGraph(structure)
        components = nx.connected_components(G)
        
        for C in components:
            if len(C) < ARGS.component_size_threshold:
                # remove all component residues from chain
                for node in C:
                    chain.detach_child(node)
            else:
                num_components += 1
                num_kept_residues += len(C)
                kept_components.append(C)
    
    if len(chain) > 0:
        if ARGS.save_separate_components:
            io.set_structure(structure)
            count = 1
            for C in kept_components:
                select = SelectComponent(C)
                fname = "{}{}_{}.pdb".format(structure_id, ARGS.suffix, count)
                io.save(os.path.join(ARGS.target_directory, fname), select=select)
                count += 1
        else:
            # write whole structure to file
            io.set_structure(structure)
            io.save(os.path.join(ARGS.target_directory, fname))
    
    # write stats to file
    OUT.write("\n{},{:.3f},{},{:.3f},{}".format(
        structure_id,
        (scores < ARGS.confidence_threshold).sum()/num_residues,
        num_components,
        num_kept_residues/num_residues,
        num_residues
    ))
OUT.close()
