import numpy as np
import networkx as nx

# geobind modules
from geobind.structure import StructureData
from geobind.structure import getAtomKDTree
from geobind.structure.data import data
from geobind.structure import mapPointFeaturesToStructure, getResidueDistance

def smoothResiduePredictions(structure, residue_dict, nc=2, edge_dist_threshold=4.0, cluster_size_threshold=4):
    pos = []
    for rid in residue_dict:
        if residue_dict[rid]["label"] == 1:
            pos.append((
                rid,
                structure.get_residue(rid[3], rid[2]),
                residue_dict[rid]["class_areas"][1]
            ))
    
    # get connected components
    G = nx.Graph()
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            dist = getResidueDistance(pos[i], pos[j])
            if dist["min"] > edge_dist_threshold:
                G.add_edge(i, j)
    
    components = nx.algorithms.components.connected_components(G)
    pass # incomplete function

def vertexLabelsToResidueLabels(structure, mesh, Y, nc=2, kdt=None, id_format='dnaprodb', null_class=0, return_kdt=False, thresholds=None):    
    if isinstance(structure, list):
        atoms = structure # we were given a list of atoms
    else:
        atoms = [atom for atom in structure.get_atoms()]
    
    if kdt is None:
        kdt = getAtomKDTree(atoms)
    
    if thresholds is None:
        thresholds = data.buried_sesa_cutoffs # these are probably junk
    elif isinstance(thresholds, float):
        _t = {}
        for res in data.standard_residues:
            _t[res] = thresholds
        thresholds = _t
    
    # get vertex areas
    areas = np.zeros_like(Y, dtype=np.float32)
    np.add.at(areas, mesh.faces[:, 0], mesh.area_faces/3)
    np.add.at(areas, mesh.faces[:, 1], mesh.area_faces/3)
    np.add.at(areas, mesh.faces[:, 2], mesh.area_faces/3)
    
    for c in range(nc):
        mask = (Y == c)
        # map every vertex area to nearest atom, based on class of vertex
        mapPointFeaturesToStructure(mesh.vertices, atoms, areas*mask, 'area_{}'.format(c), kdtree=kdt)
    
    # aggregate over atom areas
    residue_dict = {}
    for atom in atoms:
        areas = np.zeros(nc)
        for c in range(nc):
            key = 'area_{}'.format(c)
            if key in atom.xtra:
                areas[c] += atom.xtra[key]
        if areas.sum() == 0:
            # atom makes no contribution, skip
            continue
        
        residue = atom.get_parent()
        residue_id = residue.get_full_id()
        
        if residue_id not in residue_dict:
            # make a new residue entry
            residue_dict[residue_id] = {
                'residue_name': residue.get_resname(),
                'class_areas': np.zeros(nc)
            }
        
        residue_dict[residue_id]['class_areas'] += areas
    
    # determine residue class
    ni = null_class
    for residue_id in residue_dict:
        residue_dict[residue_id]['class_areas'][ni] = 0 # mask out the null class area
        ci = np.argmax(residue_dict[residue_id]['class_areas'])
        resn = residue_dict[residue_id]['residue_name']
        
        if residue_dict[residue_id]['class_areas'][ci] > thresholds[resn]:
            residue_dict[residue_id]['label'] = ci
        else:
            residue_dict[residue_id]['label'] = null_class
    
    # perform smoothing
    #smoothResiduePredictions(structure, residue_dict)
    
    # check id format
    if id_format == 'dnaprodb':
        _rdict = {}
        for residue_id in residue_dict:
            _rid = '{}.{}.{}'.format(residue_id[2], residue_id[3][1], residue_id[3][2])
            _rdict[_rid] = residue_dict[residue_id]
        residue_dict = _rdict
    
    if return_kdt:
        return residue_dict, kdt
    else:
        return residue_dict
