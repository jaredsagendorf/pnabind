#!/usr/bin/env python

from Bio.PDB import PDBParser, NeighborSearch
from Bio.SeqUtils import seq1
import argparse
import os
import re
import json

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("structure_files")
arg_parser.add_argument("structure_path")
arg_parser.add_argument("regexes_file")
arg_parser.add_argument("--cutoff_distance", default=4.5, type=float)
arg_parser.add_argument("--write_residue_ids", action="store_true")
arg_parser.add_argument("--binding_site_only", action="store_true")
ARGS = arg_parser.parse_args()

def compileRegexes(obj):
    # compile regexes loaded from JSON files
    objType = type(obj)
    if(objType is dict):
        for key in obj:
            if(type(obj[key]) is str):
                obj[key] = re.compile(obj[key])
            else:
                compileRegexes(obj[key])
    elif(objType is list):
        for i in range(len(obj)):
            if(type(obj[i]) is str):
                obj[i] = re.compile(obj[i])
            else:
                compileRegexes(obj[i])

def res_or_nuc(component, container):
    resn = component.get_resname().strip()
    if regexes["DNA"]["STANDARD_NUCLEOTIDES"].search(resn):
        container["nuc"] = component
    elif regexes["RNA"]["STANDARD_NUCLEOTIDES"].search(resn):
        container["nuc"] = component
    elif regexes["PROTEIN"]["STANDARD_RESIDUES"].search(resn):
        container["res"] = component

def writeToFasta(handle, seq_label, seq, labels, ids=None):
    handle.write(">{}\n".format(seq_label))
    handle.write(seq + "\n")
    handle.write(labels + "\n")
    if ids is not None:
        handle.write(",".join(ids) + "\n")

parser = PDBParser()
with open(ARGS.regexes_file) as FILE:
    regexes = json.load(FILE)
    compileRegexes(regexes)

FILE_OUT = open("binding_sites.fasta", "w")
for struct in open(ARGS.structure_files):
    # get the monomer chain
    structure_file = struct.strip()
    pdbid, chain_id = structure_file.strip().split('.')[0].split('_')
    structure = parser.get_structure("", os.path.join(ARGS.structure_path, structure_file))
    chain = structure[0][chain_id]
    
    # get all nucleotides in structure
    nuc = []
    for residue in structure[0].get_residues():
        resn = residue.get_resname().strip()
        if regexes["DNA"]["STANDARD_NUCLEOTIDES"].search(resn) or regexes["RNA"]["STANDARD_NUCLEOTIDES"].search(resn):
            nuc += list(filter(lambda x: x.element != 'H', residue.get_atoms()))
    
    # search for nucleotide-residue pairs
    atom_list = list(filter(lambda x: x.element != 'H', chain.get_atoms())) + nuc
    knn = NeighborSearch(atom_list)
    pairs = knn.search_all(ARGS.cutoff_distance, level="R")
    res_ind = set()
    for p in pairs:
        pair = {
            "res": None,
            "nuc": None
        }
        res_or_nuc(p[0], pair)
        res_or_nuc(p[1], pair)
        
        if pair["res"] and pair["nuc"]:
            res_ind.add(chain.child_list.index(pair["res"]))
    
    # get min and max residue index
    res_ind = list(res_ind)
    res_ind.sort()
    if len(res_ind) < 5:
        print("Not enough binding residues found: %s_%s" % (pdbid, chain_id))
        continue
    
    # get binding labels
    labels = ["0"]*len(chain.child_list)
    for i in res_ind:
        labels[i] = "1"
    labels = "".join(labels)
    
    # get residue ids
    res_ids = None
    if ARGS.write_residue_ids:
        res_ids = ["{}.{}.{}".format(res.get_parent().id, res.id[1], res.id[2]) for res in chain]
    
    # extract binding site sequence
    sequence = seq1(''.join(residue.resname if residue.id[0] == ' ' else "" for residue in chain))
    
    if ARGS.binding_site_only:
        imin = res_ind[0]
        imax = res_ind[-1]
        
        sequence = sequence[imin:imax+1]
        labels = labels[imin:imax+1]
        res_ids = res_ids[imin:imax+1]
    
    # write to file
    seq_label = "%s_%s" % (pdbid, chain_id)
    writeToFasta(FILE_OUT, seq_label, sequence, labels, res_ids)
FILE_OUT.close()
