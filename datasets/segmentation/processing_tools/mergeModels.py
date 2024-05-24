#!/usr/bin/env python
import os
import sys
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Model import Model

available_chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
parser = PDBParser(QUIET=True)
io = PDBIO()

# load the structure
structure = parser.get_structure("", sys.argv[1])
ent = os.path.basename(sys.argv[1])

chains = []
# get all chains in all models
for model in structure:
    for chain in model:
        cid = chain.id
        if cid in available_chain_ids:
            available_chain_ids.remove(cid)
        chain.detach_parent()
        chains.append(chain)

# create new model
used_ids = set()
model = Model(0)
for chain in chains:
    cid = chain.id
    if cid in used_ids:
        new_id = available_chain_ids.pop()
        chain.id = new_id
    else:
        used_ids.add(cid)
    
    model.add(chain)

# write model to file
io.set_structure(model)
io.save(ent)
