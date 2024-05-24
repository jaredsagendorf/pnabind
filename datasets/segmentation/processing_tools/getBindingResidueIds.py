#!/usr/bin/env python
import os
import sys
import json
import argparse
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("structure_path")
arg_parser.add_argument("fasta_file")
arg_parser.add_argument("chain_info_file")
arg_parser.add_argument("-m", "--chain_map", help="chain mapping file")
arg_parser.add_argument("-n", "--num_lines", default=3, type=int)
arg_parser.add_argument("--csv", dest="json", action="store_false")
arg_parser.add_argument("--sep_char", default="")
ARGS = arg_parser.parse_args()

BLOSUM = substitution_matrices.load("BLOSUM62")
parser = PDBParser(QUIET=True)

def readFASTA(file_name):
    # read in a FASTA file that contains multiple sequences. Each sequence followed
    # by a list of residue labels
    lines = open(file_name).readlines()
    i = 0
    headers = []
    seqs = []
    labels = []
    while True:
        j = ARGS.num_lines*i
        if j >= len(lines):
            break
        headers.append(lines[j].strip('\n')[1:])
        seqs.append(lines[j+1].strip('\n'))
        labels.append(list(lines[j+2].strip('\n')))
        
        i += 1
    
    return zip(headers, seqs, labels)

def readChainMappings(file_name, json_file=True):
    raise Exception("readChaiMappings clobbers all but first target chain encountered in JSON file! Probably need to use chain_info.txt file as well.")
    chain_to_chain = {}
    if json_file:
        with open(file_name) as FH:
            mappings = json.load(FH)
        # read in mappings from biol. assemb. chain id to asym. unit chain id
        for struct_id in mappings:
            pdb_id, _ = struct_id.split("_")
            asym_to_biol = {}
            for biol_id in mappings[struct_id]:
                asym_id = mappings[struct_id][biol_id]
                if asym_id not in asym_to_biol:
                    asym_to_biol[asym_id] = set()
                asym_to_biol[asym_id].add(biol_id)
            
            for asym_id in asym_to_biol:
                chain_to_chain[pdb_id+asym_id] = (struct_id, list(asym_to_biol[asym_id]))
    else:
        # read pairwise chain mappings
        for line in open(file_name):
            target, source = line.strip().split(',')
            tar_id, tar_chain_id = target.split('_')
            src_id, src_chain_id = source.split('_')
            
            chain_to_chain[src_id+src_chain_id] = ("%s.pdb" % tar_id, [tar_chain_id])
    
    return chain_to_chain

def getChainStructureMapping:
    with open(ARGS.chain_info_file) as FH:
        next(FH) # skip header
        for line in FH:
            line = line.split()
            structure_file = line[0]
            pdb_id = line[1]
            if ARGS.sep_char == "":
                target_chains = list(line[2])
                nontarget_chains = list(line[3])
            else:
                target_chains = line[2].split(ARGS.sep_char)
                nontarget_chains = line[3].split(ARGS.sep_char)
            target_chains = set(target_chains)
            nontarget_chains = set(nontarget_chains)
            nontarget_chains.discard("-")
            
            for chain_id in list(target_chains):
                target_chains.update(chain_mappings['%s_%s' %(pdb_id, chain_id)][1])
            mask_chains = nontarget_chains - target_chains
            mask[structure_file] = list(mask_chains)
CHAIN_FILE_MAPPING = readChainMappings(ARGS.chain_map, json_file=ARGS.json)
print(CHAIN_FILE_MAPPING)
exit(0)
BINDING_IDS = {}
for structure_id, polym_sequence, labels in readFASTA(ARGS.fasta_file):
    # Get chain object
    pdb_id, chain_id = structure_id.split('_')
    key = pdb_id + chain_id
    if key in CHAIN_FILE_MAPPING:
        structure_file, chain_ids = CHAIN_FILE_MAPPING[key]
    else:
        print("Structure/chain not found: %s" % key)
        continue
    structure = parser.get_structure("", os.path.join(ARGS.structure_path, structure_file))
    chain = structure[0][chain_ids[0]]
    
    # Get chain sequence
    residues = [residue for residue in chain]
    chain_sequence = seq1(''.join([residue.resname for residue in residues]))
    
    # Get chain sequence and polymer sequence alignment
    a = pairwise2.align.globalds(polym_sequence, chain_sequence, BLOSUM, -5.0, -0.5, one_alignment_only=True, penalize_end_gaps=False, penalize_extend_when_opening=True)
    seqp, matches, seqc = format_alignment(*a[0]).split("\n")[0:3]
    
    # Get residue ids corresponding to aligned labels
    ip = 0
    ic = 0
    mapp = [None]*len(polym_sequence) # corresponding point in seq p
    for i in range(len(matches)):
        if seqp[i] == '-':
            ic += 1
        
        if seqc[i] == '-':
            ip += 1
        
        if matches[i] == '|' or matches[i] == '.':
            mapp[ip] = residues[ic].get_id()
            ic += 1
            ip += 1
    
    # Write out positive labels
    residue_ids = []
    for chain_id in chain_ids:
        for i in range(len(mapp)):
            if labels[i] == "1" and mapp[i] is not None:
                res_id = "{}.{}.{}".format(chain_id, mapp[i][1], mapp[i][2])
                residue_ids.append(res_id)
    if structure_file not in BINDING_IDS:
        BINDING_IDS[structure_file] = set()
    BINDING_IDS[structure_file].update(residue_ids)

for key in BINDING_IDS:
    BINDING_IDS[key] = list(sorted(BINDING_IDS[key]))

OUT = open("binding_residue_ids.json", "w")
OUT.write(json.dumps(BINDING_IDS, indent=2))
OUT.close()
