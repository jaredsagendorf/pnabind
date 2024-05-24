import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("chain_info_file")
arg_parser.add_argument("--chain_mapping_file")
arg_parser.add_argument("--sep_char", default="")
ARGS = arg_parser.parse_args()

import sys
import json

def readChainMappings(file_name):
    with open(file_name) as FH:
        mappings = json.load(FH)
    
    chain_to_struct = {}
    for struct_id in mappings:
        pdb_id, _ = struct_id.split("_")
        asym_to_biol = {}
        for biol_id in mappings[struct_id]:
            asym_id = mappings[struct_id][biol_id]
            if asym_id not in asym_to_biol:
                asym_to_biol[asym_id] = set()
            asym_to_biol[asym_id].add(biol_id)
        
        for asym_id in asym_to_biol:
            chain_to_struct["%s_%s" % (pdb_id, asym_id)] = (struct_id, list(asym_to_biol[asym_id]))
    
    return chain_to_struct

if ARGS.chain_mapping_file:
    chain_mappings = readChainMappings(ARGS.chain_mapping_file)
mask = {}
with open(ARGS.chain_info_file) as FH:
    next(FH)
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
        if ARGS.chain_mapping_file:
            for chain_id in list(target_chains):
                target_chains.update(chain_mappings['%s_%s' %(pdb_id, chain_id)][1])
        mask_chains = nontarget_chains - target_chains
        mask[structure_file] = list(mask_chains)

OUT = open("masked_chains.json", "w")
OUT.write(json.dumps(mask, indent=2))
OUT.close()
