#!/usr/bin/env python
from .get_atom_kdtree import getAtomKDTree

def pruneStructure(structure, target_chains, threshold=30.0, size_limit=45000):
    target_atoms = []
    prune_chains = []
    total_size = 0
    for chain in structure.get_chains():
        for residue in chain:
            total_size += len(residue)
        cid = chain.get_id()
        if cid in target_chains:
            target_atoms += list(chain.get_atoms())
        else:
            prune_chains.append(chain)
    if total_size < size_limit:
        # don't do any pruning
        return
    
    # prune chains
    kdt = getAtomKDTree(target_atoms)
    for chain in prune_chains:
        remove = set()
        for atom in chain.get_atoms():
            d, i = kdt.query(atom.coord)
            if d > threshold:
                # remove residue
                remove.add(atom.get_parent())
        for res in remove:
            chain.detach_child(res.get_id())
