#!/usr/bin/env python

import sys
import wget
from tqdm import tqdm
import os

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("uniprot_mapping_file", help="map from chain id to uniprot id")
arg_parser.add_argument("-o", "--output_file", default="found.txt")
arg_parser.add_argument("-r", "--rename", choices=["bound_id", "unbound_id"], default="bound_id")
ARGS = arg_parser.parse_args()

found = open(ARGS.output_file,"w")
for line in tqdm([l.strip() for l in open(ARGS.uniprot_mapping_file,'r').readlines()]):
    chain_id, uniprot_id = line.split(",")
    if uniprot_id  == "-":
        continue
    fname = "AF-%s-F1-model_v4.pdb" % uniprot_id
    url = "https://alphafold.ebi.ac.uk/files/%s" % fname
    
    try: 
        fname = wget.download(url)
        if ARGS.rename == "bound_id":
            os.rename(fname, "%s.pdb" % chain_id)
        else:
            os.rename(fname, "%s_A.pdb" % uniprot_id)
        found.write("%s\n" % chain_id)
    except Exception as e:
        continue
found.close()
