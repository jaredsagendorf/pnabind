#!/usr/bin/env python

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("chain_file", help="list of PDB and chain ids and their corresponding assemblies")
arg_parser.add_argument("structures_directory", help="directory where structures are stored")
arg_parser.add_argument("output_directory", help="directory where outputs are written")
arg_parser.add_argument("--chain_info_file", default="chain_info.txt")
arg_parser.add_argument("--format_spec", default="{pdb_id}_{target_chains}", help="file name spec")
arg_parser.add_argument("--keep_nucleotide_chains", action='store_true', help="keep all nucleotide chains found in assembly")
arg_parser.add_argument("--only_target_nucleotide_contacts", action='store_true', help="only keep nucleotide chains contacting target protein chains")
arg_parser.add_argument("--max_residue_distance", default=4.5, type=float, help="distance threshold for protein-chain cutoffs")
arg_parser.add_argument("--nucleotide_distance", default=7.0, type=float, help="distance threshold for nucleotide-chain cutoffs")
arg_parser.add_argument("--chain_min_neighbors_count", default=5, type=int, help="minimum number of residues that meet distance threshold to count two chains as interacting")
arg_parser.add_argument("--sep_char", default="", help="chain ID separation character")
arg_parser.add_argument("--assembly_file_format", default="pdb", choices=["cif", "pdb"], help="file format of biological assemblies")
arg_parser.add_argument("--append_chain_info", action="store_true", help="append instead of overwrite chain info")
arg_parser.add_argument("--single_protein_chain", action="store_true", help="don't try to find neighboring protein chains")
arg_parser.add_argument("--no_merge", action="store_false", dest="merge", default=True,  help="don't merge multiple models within a structure")
arg_parser.add_argument("--split_large_structures", action="store_true", help="store very large structures as single-chains")
arg_parser.add_argument("--split_structure_threshold", type=int, default=15000, help="large structure split threshold")
arg_parser.add_argument("--remove_disconnected_components", action="store_true", help="remove disconnected regions not contacting nucleic acids")
arg_parser.add_argument("--asymm_chain_ids", dest="auth_chains", action="store_false", help="use asymmetric unit chain ids instead of author chain ids when parsing structure files")
arg_parser.add_argument("--lower_case_pdbid", action="store_true", help="change case of PDB identifiers to lower case")
arg_parser.add_argument("--min_chain_length", type=int, default=25, help="minimum length of chain to keep")
arg_parser.add_argument("--rename_chain_ids", action="store_true", help="rename chain ids to single character")
arg_parser.add_argument("--prune_nucleic_chains", action="store_true", help="remove nucleotides that are beyond threshold distance from any protein chain")
ARGS = arg_parser.parse_args()

import os
import sys
import json
import numpy as np
from copy import deepcopy
from Bio.PDB import PDBParser, PDBIO, FastMMCIFParser, MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB import NeighborSearch
from Bio.PDB.PDBIO import Select
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB import parse_pdb_header
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import networkx as nx
from glob import glob
from tqdm import tqdm
from scipy.spatial import cKDTree

# geobind imports
from geobind.structure.data import data as D

class SelectChains(Select):
    def __init__(self, chains):
        self.chains = chains
    
    def accept_chain(self, chain):
        return chain.get_id() in self.chains

class assembly_operation:
    def __init__(self, op_id, op_type, rotation, translation):
        self.op_id = op_id
        self.op_type = op_type
        self.rotation = np.array(rotation,dtype=np.float32).reshape(3,3)
        self.translation = np.array(translation,dtype=np.float32)

def stripHydrogens(structure):
    """Strip all hydrogen atoms from the given model.
    
    Parameters
    ----------
    model: BioPython MODEL object
        The model to be stripped of hydrogens.
    """
    if structure.get_level() == 'R':
        rm = []
        for atom in structure:
            if atom.element == 'H':
                rm.append(atom.get_id())
        for aid in rm:
            structure.detach_child(aid)
    else:
        for residue in structure.get_residues():
            rm = []
            for atom in residue:
                if atom.element == 'H':
                    rm.append(atom.get_id())
            for aid in rm:
                residue.detach_child(aid)

def nucleotideChain(chain):
    nucleotides = set(['DA', 'DC', 'DG', 'DT', 'DU', 'DI', 'A', 'C', 'G', 'T', 'U', 'I'])
    
    nuc_count = 0
    for residue in chain:
        resn = residue.get_resname().strip()
        if resn in nucleotides:
            nuc_count += 1
    
    return nuc_count/len(chain) > 0.1

def removeHETATM(chain):
    # remove any non-DNA/RNA or non-protein residue, allowing for chemical modifications of either
    remove = []
    for residue in chain:
        resname = residue.get_resname().strip()
        
        if (resname in D.standard_residues) or (resname in D.standard_RNA_nucleotides) or resname in (D.standard_DNA_nucleotides):
            continue
        else:
            if resname in D.chem_components and '_chem_comp.mon_nstd_parent_comp_id' in D.chem_components:
                parent = D.chem_components[resname]['_chem_comp.mon_nstd_parent_comp_id']
                if (parent in D.standard_residues) or (parent in D.standard_RNA_nucleotides) or (parent in D.standard_DNA_nucleotides):
                    continue
        
        remove.append(residue.get_id())
    
    for residue_id in remove:
        chain.detach_child(residue_id)

def renameChains(structure, chain_set):
    available_chain_ids = list(reversed(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")))
    chains = []
    # get all chains in structure
    for chain in structure.get_chains():
        cid = chain.id
        chain.detach_parent()
        if cid in available_chain_ids:
            available_chain_ids.remove(cid)
        if cid not in chain_set:
            continue
        chains.append(chain)
    
    chain_id_map = {}
    model = Model(0)
    for chain in chains:
        cid = chain.id
        if len(cid) > 1:
            try:
                new_id = available_chain_ids.pop()
            except IndexError:
                raise IndexError("Biological assembly contains too many chains")
            chain.id = new_id
        else:
            new_id = cid
        
        chain_id_map[new_id] = cid
        model.add(chain)
    
    return model, chain_id_map

def mergeModels(structure, max_length=None):
    
    if ( max_length is not None ) and ( len(structure) > max_length ):
        # this is large assembly - just take asymmetric unit
        return structure, {chain.get_id(): chain.get_id() for chain in structure.get_chains()}
    
    available_chain_ids = list(reversed(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")))
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
    chain_id_map = {}
    used_ids = set()
    model = Model(0)
    for chain in chains:
        cid = chain.id
        if cid in used_ids:
            try:
                new_id = available_chain_ids.pop()
            except IndexError:
                raise IndexError("Biological assembly contains too many chains")
            chain.id = new_id
        else:
            new_id = cid
            used_ids.add(cid)
        
        chain_id_map[new_id] = cid
        model.add(chain)
    structure = Structure(structure.get_id())
    structure.add(model)
    
    return structure, chain_id_map

def loadAssembly(file_name, parser, directory=".", extension="cif", merge=True, rename_chain_ids=False):
    # load structure
    file_path = os.path.join(directory, file_name)
    try:
        structure = parser.get_structure(file_name, file_path)
    except PDBConstructionException as e:
        # at least one assembly has issues - skip this structure and follow up manually
        ERR.write("{}\t{}\n".format(file_name, str(e)))
        return None, None
    
    # remove HETATOMS because occasionally a chain is HETATM only, causing issues with chain mapping
    for model in structure:
        remove = []
        for chain in model:
            removeHETATM(chain)
            if len(chain) == 0:
                # the chain only contained HETATMS and should be removed
                remove.append(chain.get_id())
        
        # remove empty chains
        for cid in remove:
            model.detach_child(cid)
    
    # determine structure type
    if extension == "cif":
        header_info = MMCIF2Dict(file_path)
        is_crystal = (header_info['_exptl.method'][0] == 'X-RAY DIFFRACTION')
    else:
        header_info = parse_pdb_header(file_path)
        is_crystal = (header_info['structure_method'] == 'x-ray diffraction')
        
    # combine symmetry related models by re-assigning chain ids
    if merge and is_crystal and len(structure) > 1:
        try:
            structure, chain_id_map = mergeModels(structure, max_length=5)
        except IndexError as e:
            # at least one assembly is too large - follow up manually
            ERR.write("{}\t{}\t{}\n".format(file_name, str(e)))
            return None, None
    else:
        chain_id_map = {chain.get_id(): chain.get_id() for chain in structure.get_chains()}
    structure = structure[0]
    
    # get rid of hydrogen atoms
    stripHydrogens(structure)
    
    return structure, chain_id_map

def getNeighboringNucleotideChains(nucleic_chains, protein_chains=None, atoms=None, distance=5.0):
    # Find all neighboring NA chains to protein chains
    if atoms is None:
        atoms = []
        for p in protein_chains:
            atoms += protein_chains[p]
    
    for n in nucleic_chains:
        atoms += nucleic_chains[n]
    
    kdt = NeighborSearch(atoms)
    pairs = kdt.search_all(distance, level="C")
    nuc_chains = set()
    for p in pairs:
        chain0 = p[0].get_id()
        chain1 = p[1].get_id()
        
        is_nuc0 = ( chain0 in nucleic_chains )
        is_nuc1 = ( chain1 in nucleic_chains )
        
        if not ( is_nuc0 ^ is_nuc1 ):
            continue
            
        if is_nuc0:
            nuc_chains.add(chain0)
        if is_nuc1:
            nuc_chains.add(chain1)
    
    return nuc_chains

def makeResidueGraph(structure, chain_set, distance_cutoff=4.5):
    alist = []
    G = nx.Graph()
    for chain in structure.get_chains():
        if chain.get_id() in chain_set:
            alist += list(chain.get_atoms())
            # add graph nodes
            for residue in chain:
                G.add_node(residue.get_full_id())
    kdt = NeighborSearch(alist)
    
    # add graph edges
    pairs = kdt.search_all(distance_cutoff, level='R')
    for pair in pairs:
        G.add_edge(pair[0].get_full_id(), pair[1].get_full_id())
    
    return G

def removeDisconnectedComponents(structure, target_chains, nontarget_chains=None, patom_dict=None, natom_dict=None, check_nucleotide_contacts=True, **kwargs):
    if nontarget_chains is None:
        nontarget_chains = set()
    G = makeResidueGraph(structure, target_chains.union(nontarget_chains), **kwargs)
    
    components = list(nx.connected_components(G))
    if len(components) == 1:
        # no op
        return target_chains, nontarget_chains
    
    # choose component with largest number of target chains/residues
    component_info = []
    for i, C in enumerate(components):
        if check_nucleotide_contacts and natom_dict is not None:
            atoms = []
            for r in C:
                atoms += list(structure[r[2]][r[3]].get_atoms())
            nuc_chains = getNeighboringNucleotideChains(natom_dict, atoms=atoms, distance=ARGS.nucleotide_distance)
            num_nuc = int(len(nuc_chains) > 0)
        else:
            num_nuc = 0
        
        num_target_chains = len(set([r[2] for r in C]).intersection(target_chains))
        num_residues = len(C)
        
        component_info.append((num_nuc, num_target_chains, num_residues, i))
    component_info.sort(reverse=True)
    keep_component = component_info[0][3]
    
    # remove all residues outside this component
    for i, C in enumerate(components):
        if i == keep_component:
            continue
        
        for r in C:
            chain = structure[r[2]]
            chain.detach_child(r[3])
    
    # remove empty chains from structure and sets
    remove = []
    for chain in structure.get_chains():
        if len(chain) == 0:
            cid = chain.get_id()
            remove.append(cid)
            target_chains.discard(cid)
            nontarget_chains.discard(cid)
            if patom_dict is not None:
                del patom_dict[cid]
    for cid in remove:
        structure.detach_child(cid)
        
    # remove residues with no parent
    if patom_dict is not None:
        for cid in patom_dict:
            patom_dict[cid] = list(filter(lambda x: x.get_parent().get_parent() is not None, patom_dict[cid]))
    
    return target_chains, nontarget_chains

def checkChainLength(structure, nontarget_chains, min_length):
    # remove non-target chains shorter than minimum
    remove = []
    for cid in list(nontarget_chains):
        if len(structure[cid]) < min_length:
            remove.append(cid)
            nontarget_chains.discard(cid)
    for cid in remove:
        structure.detach_child(cid)

def pruneStructure(structure, target_chains, threshold=30.0, size_limit=75000):
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
    coords = np.array([atom.coord for atom in target_atoms])
    kdt = cKDTree(coords)
    for chain in prune_chains:
        remove = set()
        for atom in chain.get_atoms():
            d, i = kdt.query(atom.coord)
            if d > threshold:
                # remove residue
                remove.add(atom.get_parent())
        for res in remove:
            chain.detach_child(res.get_id())

def saveFile(structure, pdb_id, target_chains, nontarget_chains, nucleic_chains, natom_dict=None, chain_mappings=None, rename_chains=False, prune_nucleotides=False):
    def _remapSet(chain_ids, mappings):
        new_chain_ids = set()
        for cid in chain_ids:
            new_chain_ids.add(mappings[cid])
        
        return new_chain_ids
    
    structure = deepcopy(structure)
    if ARGS.remove_disconnected_components:
        target_chains, nontarget_chains = removeDisconnectedComponents(structure, target_chains, nontarget_chains, natom_dict=natom_dict, distance_cutoff=ARGS.max_residue_distance)
    checkChainLength(structure, nontarget_chains, ARGS.min_chain_length)
    
    chain_set = target_chains.union(nontarget_chains).union(nucleic_chains)
    pt_chains = ARGS.sep_char.join(sorted(target_chains)) if len(target_chains) else '-' 
    pn_chains = ARGS.sep_char.join(sorted(nontarget_chains)) if len(nontarget_chains) else '-'
    nn_chains = ARGS.sep_char.join(sorted(nucleic_chains)) if len(nucleic_chains) else '-'
    fname = ARGS.format_spec.format(pdb_id=pdb_id, target_chains=pt_chains, other_chains=pn_chains, na_chains=nn_chains)
    fname = "%s.%s" % (fname, output_file_format)
    
    if rename_chains:
        # only useful for structures in CIF format which have chain ids
        # longer than one character
        structure, chain_mappings = renameChains(structure, chain_set)
        invert_mappings = {}
        for new, old in chain_mappings.items():
            invert_mappings[old] = new
        
        target_chains = _remapSet(target_chains, invert_mappings)
        nontarget_chains = _remapSet(nontarget_chains, invert_mappings)
        nucleic_chains = _remapSet(nucleic_chains, invert_mappings)
        chain_set = _remapSet(chain_set, invert_mappings)
        pt_chains = ARGS.sep_char.join(sorted(target_chains)) if len(target_chains) else '-' 
        pn_chains = ARGS.sep_char.join(sorted(nontarget_chains)) if len(nontarget_chains) else '-'
        nn_chains = ARGS.sep_char.join(sorted(nucleic_chains)) if len(nucleic_chains) else '-'
    
    if prune_nucleotides:
        pruneStructure(structure, target_chains.union(nontarget_chains))
    # save structure
    io.set_structure(structure)
    select = SelectChains(chain_set)
    io.save(os.path.join(ARGS.output_directory, fname), select=select)
    
    # write out chain info
    CHAIN_INFO_FILE.write("\n{}\t{}\t{}\t{}\t{}".format(
        fname,
        pdb_id,
        pt_chains,
        pn_chains,
        nn_chains
    ))
    
    if chain_mappings is not None:
        # keep track of asym_id -> assemb_id chain mappings
        CHAIN_MAPPINGS[fname] = {}
        for c in chain_set:
            CHAIN_MAPPINGS[fname][c] = chain_mappings[c]

def processSingleChain(chain_id, chain, nucleic_chains, structure, pdb_id, **kwargs):
    # get chain sets
    chain_set = set([chain_id]) # protein + nucleic
    target_chains = set([chain_id]) # target protein
    non_targets = set() # non-target protein (empty)
    
    if ARGS.keep_nucleotide_chains:
        # get nucleotide chains
        pchains = {chain_id: chain}
        nuc_chains = getNeighboringNucleotideChains(nucleic_chains, pchains, distance=ARGS.nucleotide_distance)
        chain_set.update(nuc_chains)
    else:
        nuc_chains = set()
    
    # Write to file
    saveFile(structure, pdb_id, target_chains, non_targets, nuc_chains, natom_dict=nucleic_chains, rename_chains=ARGS.rename_chain_ids, prune_nucleotides=ARGS.prune_nucleic_chains, **kwargs)

# I/O classes
if ARGS.assembly_file_format == "cif":
    parser = MMCIFParser(QUIET=True, auth_chains=ARGS.auth_chains)
    io = MMCIFIO()
    ARGS.merge = False
    output_file_format = "cif"
else:
    parser = PDBParser(QUIET=True)
    io = PDBIO()
    output_file_format = "pdb"

if ARGS.rename_chain_ids:
    io = PDBIO()
    output_file_format = "pdb"

# Read in chain IDs/assembly files
CHAIN_IDS = {} # keyed by <pdbid> -> [(chain_id, assembly_file)]
for line in open(ARGS.chain_file):
    line = line.strip().split(',')
    pdb_id, chain_id = line[0].split('_')
    assembly_file = line[1]
    if ARGS.lower_case_pdbid:
        pdb_id = pdb_id.lower()
    
    if pdb_id not in CHAIN_IDS:
        CHAIN_IDS[pdb_id] = {}
    if assembly_file not in CHAIN_IDS[pdb_id]:
        CHAIN_IDS[pdb_id][assembly_file] = []
    
    CHAIN_IDS[pdb_id][assembly_file].append(chain_id)

# Output files
if os.path.exists(ARGS.chain_info_file) and ARGS.append_chain_info:
    CHAIN_INFO_FILE = open(ARGS.chain_info_file, "a")
else:
    CHAIN_INFO_FILE = open(ARGS.chain_info_file, "w")
    CHAIN_INFO_FILE.write("file_name\tpdb_identifier\ttarget_chain_id\tneighbor_chain_ids\tna_chain_ids")
ERR = open("error_structures.txt", "w") # structures which couldn't be parsed or not found

# Read in structures
CHAIN_MAPPINGS = {}
for pdb_id in tqdm(CHAIN_IDS, desc=" Structure", position=0):
    # load assemblies
    for assembly_file in CHAIN_IDS[pdb_id]:
        structure, chain_mappings = loadAssembly(assembly_file, parser, ARGS.structures_directory, extension=ARGS.assembly_file_format, merge=ARGS.merge)
        if structure is None:
            # couldn't parse file
            continue
        
        # Get all chains in structure
        protein_chains = {}
        nucleic_chains = {}
        for chain in structure.get_chains():
            # check if chain is NA or Protein
            na = nucleotideChain(chain)
            atoms = list(chain.get_atoms())
            cid = chain.get_id()
            if na:
                nucleic_chains[cid] = atoms
            else:
                protein_chains[cid] = atoms
        
        # Decide which chains to write 
        if ARGS.single_protein_chain:
            for chain_id in CHAIN_IDS[pdb_id][assembly_file]:
                # if ARGS.remove_disconnected_components:
                    # removeDisconnectedComponents(structure, set(chain_id), nucleic_chains=nucleic_chains, patom_dict=protein_chains, distance_cutoff=ARGS.max_residue_distance)
                processSingleChain(
                    chain_id,
                    protein_chains[chain_id],
                    nucleic_chains,
                    structure,
                    pdb_id,
                    chain_mappings=chain_mappings
                )
        else:
            # get pairs of protein residues within cutoff distance of each target chain
            patom_list = []
            for a in protein_chains.values():
                patom_list += a
            pkdt = NeighborSearch(patom_list)
            pairs = pkdt.search_all(ARGS.max_residue_distance, level="R")
            
            # get edges
            edge_counts = {}
            for p in pairs:
                chain0 = p[0].get_parent().get_id()
                chain1 = p[1].get_parent().get_id()
                
                edge = tuple(sorted([chain0, chain1]))
                if edge not in edge_counts:
                    edge_counts[edge] = 0
                edge_counts[edge] += 1
            
            # add edges that meet threshold
            G = nx.Graph()
            for edge in edge_counts:
                if edge_counts[edge] >= ARGS.chain_min_neighbors_count:
                    G.add_edge(edge[0], edge[1])
            
            # # Loop over target chains in assembly and get all neighbors
            # for C in nx.connected_components(G):
                # # find chains connected to target chains
                # chain_set = set()
                # target_chains = set()
                # for chain_id in CHAIN_IDS[pdb_id][assembly_file]:
                    # if chain_id in C:
                        # target_chains.add(chain_id)
                        # chain_set.update(G.neighbors(chain_id))
                # nontarget_chains = chain_set - target_chains
                
                # if len(target_chains) == 0:
                    # # no target chains in this component
                    # continue
            for target_chain_id in CHAIN_IDS[pdb_id][assembly_file]:
                chain_set = set([target_chain_id])
                target_chains = set([target_chain_id])
                chain_set.update(G.neighbors(target_chain_id))
                nontarget_chains = chain_set - target_chains
                
                # determine the size of the complex and split if too large and requested
                psize = sum([len(protein_chains[c]) for c in chain_set])
                if psize > ARGS.split_structure_threshold and ARGS.split_large_structures:
                    # save each target chain individually
                    for chain_id in target_chains:
                        # if ARGS.remove_disconnected_components:
                            # removeDisconnectedComponents(structure, set(chain_id), nucleic_chains=nucleic_chains, patom_dict=protein_chains, distance_cutoff=ARGS.max_residue_distance)
                        processSingleChain(
                            chain_id,
                            protein_chains[chain_id],
                            nucleic_chains,
                            structure,
                            pdb_id,
                            chain_mappings=chain_mappings
                        )
                else:
                    # if ARGS.remove_disconnected_components:
                        # # remove disconnected regions from chain set
                        # target_chains, nontarget_chains = removeDisconnectedComponents(structure, target_chains, nontarget_chains, nucleic_chains=nucleic_chains, patom_dict=protein_chains, distance_cutoff=ARGS.max_residue_distance)
                        # chain_set = target_chains.union(nontarget_chains)
                    
                    if ARGS.keep_nucleotide_chains:
                        if ARGS.only_target_nucleotide_contacts:
                            pchains = {c: protein_chains[c] for c in target_chains}
                        else:
                            pchains = {c: protein_chains[c] for c in chain_set}
                        
                        nuc_chains = getNeighboringNucleotideChains(nucleic_chains, pchains, distance=ARGS.nucleotide_distance)
                        chain_set.update(nuc_chains)
                    else:
                        nuc_chains = set()
                    
                    # Write to file
                    saveFile(structure, pdb_id, target_chains, nontarget_chains, nuc_chains, natom_dict=nucleic_chains, chain_mappings=chain_mappings, rename_chains=ARGS.rename_chain_ids, prune_nucleotides=ARGS.prune_nucleic_chains)

CHAIN_INFO_FILE.close()
ERR.close()
MAP = open("chain_mappings.json", "w")
MAP.write(json.dumps(CHAIN_MAPPINGS, indent=2))
MAP.close()
