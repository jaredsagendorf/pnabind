#!/usr/bin/env python
import logging
import argparse
from os.path import join as ospj
from glob import glob

import numpy as np
import trimesh
from Bio.PDB import PDBParser

from geobind.nn.metrics import report
from geobind.structure import getAtomKDTree, StructureData, getAtomSESA
from geobind.utils import oneHotEncode
from geobind.structure.data import data as D

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("path", help="directory where .npz files are located")
arg_parser.add_argument("-P", "--pdb_dir", default=".", help="pdb file directory prefix")
arg_parser.add_argument("-t", "--threshold", type=float, default=0.5, help="threshold for evaluating single-point metrics")
arg_parser.add_argument("-l", "--level", default="R", help="level at which to perform mapping")
arg_parser.add_argument("-o", "--output", default="log", choices=["log", "csv"])
ARGS = arg_parser.parse_args()

logging.basicConfig(format='%(levelname)s:    %(message)s', level=logging.INFO)

def mapVertexLabelsToStructure(vertices, atom_list, vertex_labels, weighting='binary', weights=None, level='R', kdtree=None, nc=None):
    # determine number of classes
    if nc is None:
        nc = np.unique(vertex_labels).size
    
    # build a KDTree for structure if one is not provided
    if kdtree is None:
        kdtree = getAtomKDTree(atom_list)
    
    for atom in atom_list:
        if('label_weight' in atom.xtra):
            atom.xtra['label_weight'] = np.zeros(nc)
            
    # find nearest-neighbor atoms and add label weight
    dist, ind = kdtree.query(vertices)
    for i in range(len(ind)):
        ai = ind[i]
        vi = i
        if(weighting == 'binary'):
            w = 1
        elif(weighting == 'biased'):
            w = weights[vertex_labels[vi]]
        
        if('label_weight' not in atom_list[ind[i]].xtra):
            atom_list[ai].xtra['label_weight'] = np.zeros(nc)
        atom_list[ai].xtra['label_weight'][vertex_labels[vi]] += w
    
    # return list of entities with aggregated discretized label
    if(level == 'A'):
        atom_dict = {}
        for atom in atom_list:
            if('label_weight' in atom.xtra):
                atom_dict[atom.get_full_id()] = np.argmax(atom.xtra['label_weight'])
        return atom_dict
    elif(level == 'R'):
        residue_dict = {}
        # aggregate over atom weights
        for atom in atom_list:
            if('label_weight' in atom.xtra):
                residue = atom.get_parent()
                residue_id = residue.get_full_id()
                if(residue_id not in residue_dict):
                    residue_dict[residue_id] = np.zeros(nc)
                residue_dict[residue_id] += atom.xtra['label_weight']
        
        # assign class to residue
        for residue_id in residue_dict:
            residue_dict[residue_id] = np.argmax(residue_dict[residue_id])
        
        
        return residue_dict

def mapVertexProbabilitiesToStructure(vertices, atom_list, P, nc, level='A', kdtree=None, vertex_weights=None, reduce_method='mean'):
    
    # one hot encode if given labels vector
    if P.ndim == 1:
        P = oneHotEncode(P, nc)
    
    # build a KDTree for structure if one is not provided
    if kdtree is None:
        kdtree = getAtomKDTree(atom_list)
    
    # reset values if atom list was used prior
    for atom in atom_list:
        atom.xtra['p'] = np.zeros(nc)
        atom.xtra['v_weight'] = 0
    
    # find nearest-neighbor atoms and add probabilities
    dist, ind = kdtree.query(vertices)
    for i in range(len(ind)):
        ai = ind[i]
        vi = i
        atom_list[ai].xtra['p'] += P[vi]*vertex_weights[vi]
        atom_list[ai].xtra['v_weight'] += vertex_weights[vi]
    
    # normalize probabilities
    for atom in atom_list:
        if atom.xtra['v_weight'] > 0:
            atom.xtra['p'] = atom.xtra['p']/atom.xtra['v_weight']
    
    # return list of entities with aggregated probabilities
    if level == 'A':
        atom_dict = {atom.get_full_id(): atom.xtra['p'] for atom in atom_list}
        return atom_dict
    elif level == 'R':
        residue_dict = {}
        # aggregate over atom probabilities
        for atom in atom_list:
            residue = atom.get_parent()
            residue_id = residue.get_full_id()
            if residue_id not in residue_dict:
                residue_dict[residue_id] = []
            residue_dict[residue_id].append(atom.xtra['p'])
        
        # reduce residue probabilities
        for residue_id in residue_dict:
            residue_dict[residue_id] = np.concatenate(residue_dict[residue_id], axis=0)
            if reduce_method == 'mean':
                residue_dict[residue_id] = np.mean(residue_dict[residue_id], axis=0)
            elif reduce_method == 'max':
                residue_dict[residue_id] = np.max(residue_dict[residue_id], axis=0)
        
        return residue_dict

def getProtein(structure, regexes, mi=0):
    """Docstring"""
    pro = []
    for chain in structure[mi]:
        for residue in chain:
            resname = residue.get_resname()
            if regexes['PROTEIN']['STANDARD_RESIDUES'].search(resname):
                pro.append(residue.get_full_id())
    
    return structure.slice(structure, pro, 'protein')

# def getLoopContent(structure, fileName):
    # getDSSP(structure[0], fileName)
    # acount = 0
    # lcount = 0
    # for atom in structure[0].get_atoms():
        # lcount += atom.xtra.get('ss_L', 0)
        # acount += 1
    
    # return lcount/acount

# def getMeanBFactor(structure):
    # acount = 0
    # bfactor = 0
    
    # for atom in structure.get_atoms():
        # acount += 1
        # bfactor += atom.bfactor
    
    # return bfactor/acount

pdb_parser = PDBParser()
use_header=True
if ARGS.output == 'csv':
    csv = open('data.csv', 'w')

for f in glob(ospj(ARGS.path, '*.npz')):
    f = f.lstrip('./')
    
    # load data file
    data = np.load(f)
    
    # load PDB structure
    pdbfile = ospj(ARGS.pdb_dir, f.rstrip("_predict.npz") + ".pdb")
    
    # apply same processing as on training data
    fname = f.rstrip("_predict.npz") + ".pdb"
    structure = StructureData(fname, name=fname, path=ARGS.pdb_dir)
    protein = getProtein(structure, D.regexes)
    
    # Clean the protein entity
    protein, pqr = geobind.structure.cleanProtein(protein hydrogens=True)
    
    # get atoms with sesa > 0
    getAtomSESA(protein)
    atoms = [atom if atom.xtra['sesa'] > 0 for atom in structure.get_atoms()]
    
    # load mesh file
    mesh = trimesh.Trimesh(vertices=data['V'], faces=data['F'], process=False, validate=False)
    
    # load vertex data
    Y = data['Y']
    P = data['P']
    Y[Y < 0] = 0 # remove mask
    
    kdt = getAtomKDTree(atoms)
    #Rd_gt = mapVertexLabelsToStructure(mesh.vertices, atoms, Ygt, kdtree=kdt, level='A')
    #Rd_pr = mapVertexLabelsToStructure(mesh.vertices, atoms, Ypr, kdtree=kdt, level='A')
    map_gt = mapVertexProbabilitiesToStructure(mesh.vertices, atoms, Ygt, 2, kdtree=kdt, level=ARGS.level, reduce_method='max')
    map_pr = mapVertexProbabilitiesToStructure(mesh.vertices, atoms, Ppr, 2, kdtree=kdt, level=ARGS.level, reduce_method='max')
    
    y = []
    p = []
    for key in map_gt:
        y.append(map_gt[key])
        p.append(map_pr[key])
    
    y = (np.array(y) >= 0.5)
    p = np.array(p)
    metrics = getMetrics(y, p, threshold=ARGS.threshold)
    
    if ARGS.output == 'csv':
        mkeys = sorted(metrics.keys())
        data = list(map(lambda key: metrics[key], mkeys))
        data += [len(mesh.vertices), len(atoms), getLoopContent(structure, pdbfile), getMeanBFactor(structure)]
        if(use_header):
            header = ['prefix'] + list(mkeys) + ['num_vertices', 'num_atoms', 'loop_content', 'mean_bfactor']
            csv.write(','.join(header))
        csv.write('\n' + ','.join([filePrefix] + ["{:.3f}".format(d) for d in data]))
    else:
        report(
            [
                ({'entity': filePrefix}, ''),
                (metrics, 'per-datum metrics')
            ],
            header=use_header
        )
    use_header=False

if ARGS.output == 'csv':
    csv.close()

