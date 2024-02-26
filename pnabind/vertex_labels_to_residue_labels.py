# third party modules
import numpy as np

# pnabind modules
from pnabind.structure import StructureData
from pnabind.structure import getAtomKDTree
from pnabind.structure.data import data
from pnabind.structure import mapPointFeaturesToStructure, getResidueDistance

def smoothResidueLabels(residues, labels, nc=2, distance_cutoff=4.0, iterations=1, skip_hydrogens=True, ignore_class=set()):
    """Apply laplacian smoothing"""
    try:
        import networkx as nx
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The dependency 'networkx' is required for this functionality!")
    
    atoms = []
    for residue in residues:
        for atom in residue:
            if skip_hydrogens and atom.element == 'H':
                continue
            atoms.append(atom)
    kdt = getAtomKDTree(atoms, engine='biopython')
    
    # get connected components
    G = nx.Graph()
    for i in range(len(residues)):
        G.add_edge(i, i) # self loop to count self
    residue_node = {residue: i for i, residue in enumerate(residues)}
    
    neighbors = kdt.search_all(distance_cutoff, level='R')
    for pair in neighbors:
        G.add_edge(
            residue_node[pair[0]],
            residue_node[pair[1]]
        )
    
    # smooth labels with simple majority vote
    for _ in range(iterations):
        new_labels = labels.copy()
        for i in range(len(labels)):
            if labels[i] in ignore_class:
                continue
            ni = list(G.neighbors(i))
            counts = np.bincount(labels[ni], minlength=nc)
            if counts[0] == counts[1]:
                # tie - leave class alone
                continue
            new_labels[i] = counts.argmax()
        labels = new_labels
    
    return labels

def vertexLabelsToResidueLabels(structure, vertices, Y,
        areas=None,
        nc=2,
        id_format='biopython',
        include_buried=False,
        kdt=None,
        return_kdt=False,
        residue_dict=None,
        store_residue=False,
        separate_backbone=False,
        **kwargs
    ):
    if isinstance(structure, list):
        atoms = structure # we were given a list of atoms
    else:
        atoms = [atom for atom in structure.get_atoms()]
    
    if kdt is None:
        kdt = getAtomKDTree(atoms)
    
    if areas is None:
        areas = np.ones_like(Y)
    
    # assign vertex areas to class areas
    for c in range(nc):
        mask = (Y == c)
        # map every vertex area to nearest atom, based on class of vertex
        mapPointFeaturesToStructure(vertices, atoms, areas*mask, 'area_{}'.format(c), kdtree=kdt, impute=True)
    
    # aggregate over atom areas
    if residue_dict is None:
        residue_dict = {}
    bb_atoms = set(['N', 'C', 'O', 'OXT'])
    for atom in atoms:
        areas = np.zeros(nc)
        for c in range(nc):
            key = 'area_{}'.format(c)
            if key in atom.xtra:
                areas[c] += atom.xtra[key]
        
        if (not include_buried) and areas.sum() == 0:
            # atom makes no contribution, skip
            continue
        
        residue = atom.get_parent()
        if id_format == "biopython":
            residue_id = residue.get_full_id()
        elif id_format == "dnaprodb":
            residue_id = residue.get_full_id()
            residue_id = '{}.{}.{}'.format(residue_id[2], residue_id[3][1], residue_id[3][2])
        else:
            residue_id = residue
        
        if residue_id not in residue_dict:
            # make a new residue entry
            residue_dict[residue_id] = {
                'residue_name': residue.get_resname(),
                'class_areas': np.zeros(nc)
            }
            if store_residue:
                residue_dict[residue_id]["residue"] = residue
            if separate_backbone:
                residue_dict[residue_id]["backbone_class_areas"] = np.zeros(nc)
        if separate_backbone and atom.name.strip() in bb_atoms:
            residue_dict[residue_id]['backbone_class_areas'] += areas
        else:
            residue_dict[residue_id]['class_areas'] += areas
        
    if return_kdt:
        return residue_dict, kdt
    else:
        return residue_dict
