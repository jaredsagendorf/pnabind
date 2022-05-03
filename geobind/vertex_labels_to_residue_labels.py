# third party modules
import numpy as np

# geobind modules
from geobind.structure import StructureData
from geobind.structure import getAtomKDTree
from geobind.structure.data import data
from geobind.structure import mapPointFeaturesToStructure, getResidueDistance

def smoothResiduePredictions(structure, residue_dict, nc=2, edge_dist_threshold=4.0, niter=1, hydrogens=False, ignore_class=set()):
    """Apply laplacian smoothing"""
    try:
        import networkx as nx
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The dependency 'networkx' is required for this functionality!")
    
    residues = []
    atoms = []
    labels = []
    count = 0
    for rid in residue_dict:
        res = structure.get_residue(rid[3], rid[2])
        labels.append(residue_dict[rid]["label"])
        for atom in res:
            if (not hydrogens) and atom.element == 'H':
                continue
            atoms.append(atom)
        residue_dict[rid]["node"] = count
        residues.append(rid)
        count += 1
    
    kdt = getAtomKDTree(atoms, engine='biopython')
    
    # get connected components
    G = nx.Graph()
    labels = np.array(labels, dtype=int)
    for i in range(count):
        G.add_edge(i, i) # self loop to count self

    neighbors = kdt.search_all(edge_dist_threshold, level='R')
    for pair in neighbors:
        G.add_edge(
            residue_dict[pair[0].get_full_id()]["node"],
            residue_dict[pair[1].get_full_id()]["node"]
        )
    
    # smooth labels with simple majority vote
    for _ in range(niter):
        new_labels = np.zeros(count)
        for i in range(count):
            if labels[i] in ignore_class:
                continue
            ni = list(G.neighbors(i))
            new_labels[i] = np.bincount(labels[ni]).argmax()
        labels = new_labels
    
    # update labels in residue_dict
    for i in range(count):
        residue_dict[residues[i]]["label"] = int(new_labels[i])

def vertexLabelsToResidueLabels(structure, mesh, Y, 
            nc=2,
            kdt=None,
            id_format='dnaprodb',
            null_class=0,
            return_kdt=False,
            thresholds=None,
            smooth_labels=False,
            **kwargs
    ):    
    if isinstance(structure, list):
        atoms = structure # we were given a list of atoms
    else:
        atoms = [atom for atom in structure.get_atoms()]
    
    if kdt is None:
        kdt = getAtomKDTree(atoms)
    
    if thresholds is None:
        thresholds = data.sesa_cutoffs # these should be used for training labels only
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
    if smooth_labels:
        smoothResiduePredictions(structure, residue_dict, nc=nc, **kwargs)
    
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
