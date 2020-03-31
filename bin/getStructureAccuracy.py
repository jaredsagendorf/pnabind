#!/usr/bin/env python
import logging
import argparse
from os.path import join as ospj

import numpy as np
import trimesh
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

from geobind.nn.utils import getMetrics, report

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("data_file", help="list of file prefixes")
arg_parser.add_argument("-M", "--mesh_dir", default=".", help="mesh file directory prefix")
arg_parser.add_argument("-P", "--pdb_dir", default=".", help="pdb file directory prefix")
arg_parser.add_argument("-L", "--label_dir", default=".", help="label file directory prefix")
arg_parser.add_argument("-t", "--threshold", type=float, default=0.5, help="threshold for evaluating single-point metrics")
arg_parser.add_argument("-l", "--level", default="R", help="level at which to perform mapping")
ARGS = arg_parser.parse_args()

logging.basicConfig(format='%(levelname)s:    %(message)s', level=logging.INFO)

def getAtomKDTree(atoms):
    coords = np.array([atom.coord for atom in atoms])
    
    return cKDTree(coords)

def mapVertexLabelsToStructure(vertices, atom_list, vertex_labels, weighting='binary', weights=None, level='R', kdtree=None, nc=None):
    # determine number of classes
    if(nc is None):
        nc = np.unique(vertex_labels).size
    
    # build a KDTree for structure if one is not provided
    if(kdtree is None):
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

def mapVertexProbabilitiesToStructure(vertices, atom_list, probabilities, level='A', kdtree=None, nc=None):
    # determine number of classes
    if(nc is None):
        if(probabilities.ndim == 1):
            nc = np.unique(probabilities).size
        else:
            nc = probabilities.shape[1]
    
    # one hot encode if given labels vector
    if(probabilities.ndim == 1):
        p = np.zeros((probabilities.size, nc))
        p[np.arange(p.shape[0]), probabilities] = 1
        probabilities = p
    
    # build a KDTree for structure if one is not provided
    if(kdtree is None):
        kdtree = getAtomKDTree(atom_list)
    
    # reset values if atom list was used prior
    for atom in atom_list:
        if('p' in atom.xtra):
            atom.xtra['p'] = np.zeros(nc)
            atom.xtra['vcount'] = 0
            
    # find nearest-neighbor atoms and add probabilities
    dist, ind = kdtree.query(vertices)
    for i in range(len(ind)):
        ai = ind[i]
        vi = i
        if('p' not in atom_list[ind[i]].xtra):
            atom_list[ai].xtra['p'] = np.zeros(nc)
            atom_list[ai].xtra['vcount'] = 0
        atom_list[ai].xtra['p'] += probabilities[vi]
        atom_list[ai].xtra['vcount'] += 1
    
    # normalize probabilities
    for atom in atom_list:
        if('p' in atom.xtra):
            atom.xtra['p'] = atom.xtra['p']/atom.xtra['vcount']
    
    # return list of entities with aggregated probabilities
    if(level == 'A'):
        atom_dict = {}
        for atom in atom_list:
            if('p' in atom.xtra):
                atom_dict[atom.get_full_id()] = atom.xtra['p']
        return atom_dict
    elif(level == 'R'):
        residue_dict = {}
        # aggregate over atom probabilities
        for atom in atom_list:
            if('p' in atom.xtra):
                residue = atom.get_parent()
                residue_id = residue.get_full_id()
                if(residue_id not in residue_dict):
                    residue_dict[residue_id] = {}
                    residue_dict[residue_id]['p'] = np.zeros(nc)
                    residue_dict[residue_id]['acount'] = 0
                residue_dict[residue_id]['p'] += atom.xtra['p']
                residue_dict[residue_id]['acount'] += 1
        
        # normalize residue probabilities
        for residue_id in residue_dict:
            residue_dict[residue_id] = residue_dict[residue_id]['p']/residue_dict[residue_id]['acount']
        
        return residue_dict

pdb_parser = PDBParser()
Y = []
P = []
use_header=True
for filePrefix in open(ARGS.data_file):
    filePrefix = filePrefix.strip()
    
    # load PDB structure
    structure = pdb_parser.get_structure('structure', ospj(ARGS.pdb_dir, filePrefix.rstrip("_protein")+".pdb"))
    atoms = [atom for atom in structure.get_atoms()]
    
    # load mesh file
    mesh = trimesh.load(ospj(ARGS.mesh_dir, filePrefix+"_mesh.off"), process=False, validate=False)
    
    # load vertex data
    Ygt = np.load(ospj(ARGS.label_dir, filePrefix+"_vertex_labels.npy"))
    #Ypr = np.load(ospj(ARGS.label_dir, filePrefix+"_vertex_labels_p.npy"))
    Ppr = np.load(ospj(ARGS.label_dir, filePrefix+"_vertex_probs.npy"))
    Ygt[Ygt < 0] = 0 # remove mask
    
    kdt = getAtomKDTree(atoms)
    #Rd_gt = mapVertexLabelsToStructure(mesh.vertices, atoms, Ygt, kdtree=kdt, level='A')
    #Rd_pr = mapVertexLabelsToStructure(mesh.vertices, atoms, Ypr, kdtree=kdt, level='A')
    map_gt = mapVertexProbabilitiesToStructure(mesh.vertices, atoms, Ygt, kdtree=kdt, level=ARGS.level)
    map_pr = mapVertexProbabilitiesToStructure(mesh.vertices, atoms, Ppr, kdtree=kdt, level=ARGS.level)
    
    y = []
    p = []
    for key in map_gt:
        y.append(map_gt[key])
        p.append(map_pr[key])
    Y += y
    P += p
    
    y = (np.array(y)[:,1] >= 0.5)
    metrics = getMetrics(y, p, threshold=ARGS.threshold)
    report(
        [
            ({'entity': filePrefix}, ''),
            (metrics, 'per-datum metrics')
        ],
        header=use_header
    )
    use_header=False

Y = (np.array(Y)[:,1] >= 0.5)
P = np.array(P)
metrics = getMetrics(Y, P, threshold=ARGS.threshold)
report([(metrics, 'summary')], header=True)
