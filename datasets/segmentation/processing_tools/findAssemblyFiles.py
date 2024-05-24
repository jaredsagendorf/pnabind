"""Read in a list of structures/chain ids and find, in a given directory,
which assembly files contain the given chain(s). If the same chain is 
found in multiple assemblies, then the user should manually decide which assembly
they want to use.

input: list in form '<pdb id>_<chain id>\n'

output: list in form '<pdb_id>_<chain_id>, <assembly 1>, <assembly 2>,...
"""

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("chain_file", help="list of PDB and chain ids")
arg_parser.add_argument("structures_directory", help="directory where structures are stored")
arg_parser.add_argument("--assembly_file_format", default="pdb", choices=["cif", "pdb"], help="file format of biological assemblies")
ARGS = arg_parser.parse_args()

import os
from glob import glob
from Bio.PDB import PDBParser, FastMMCIFParser, MMCIFParser

# Global file handles
ERR = open("error_assembly_files.txt", "w") # structures which couldn't be parsed
OUT = open("chain_assemblies.txt", "w")

def loadAssemblies(pdb_id, parser, directory=".", extension="cif"):
    if extension == "pdb":
        files = glob("%s/%s.%s*" % (directory, pdb_id, extension))
    else:
        files = glob("%s/%s-assembly*.%s" % (directory, pdb_id, extension))
    assemblies = {}
    
    # load structures
    structures = []
    for f in files:
        try:
            structure = parser.get_structure(f, f)
        except PDBConstructionException as e:
            # at least one assembly has issues - skip this structure and follow up manually
            ERR.write("{}\t{}\t{}\n".format(pdb_id, f, str(e)))
            continue 
        structures.append((structure, f))
    
    # loop over structures
    for structure, f in structures:
        f = os.path.basename(f)
        # determine chain mappings
        for chain in structure.get_chains():
            cid = chain.get_id()
            if cid not in assemblies:
                assemblies[cid] = set()
            assemblies[cid].add(f)
    
    return assemblies

# I/O classes
if ARGS.assembly_file_format == "cif":
    parser = FastMMCIFParser(QUIET=True)
else:
    parser = PDBParser(QUIET=True)

# Read in chain IDs
CHAIN_IDS = {} # keyed by <pdbid> -> [list, of, chain, ids]
for line in open(ARGS.chain_file):
    pdb_id, chain_id = line.strip().split('_')
    pdb_id = pdb_id.lower()
    if pdb_id not in CHAIN_IDS:
        CHAIN_IDS[pdb_id] = []
    CHAIN_IDS[pdb_id].append(chain_id)

# Search for assemblies containing each chain
OUT.write("# The user should manually check this file for multiple assemblies containing the same chain and edit to select one of them\n")
for pdb_id in CHAIN_IDS:
    assemblies = loadAssemblies(pdb_id, parser, ARGS.structures_directory, extension=ARGS.assembly_file_format)
    
    for chain_id in CHAIN_IDS[pdb_id]:
        if chain_id in assemblies:
            OUT.write("%s_%s," % (pdb_id, chain_id) + ",".join(sorted(list(assemblies[chain_id]))) + "\n")
        else:
            print("assembly not found: %s_%s" % (pdb_id, chain_id))

ERR.close()
OUT.close()
