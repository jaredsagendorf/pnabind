import numpy as np
from .get_atom_kdtree import getAtomKDTree

def getAngle(v1, v2):
    cp = np.cross(v1, v2, axis=-1)
    dp = (v1*v2).sum(axis=-1)
    
    return np.arctan2(np.linalg.norm(cp, axis=-1), dp)

def getEffectiveInteractions(X, Y, omega):
    atoms = np.array(Y)
    X = X.coord
    Y = np.array([y.coord for y in Y])
    
    # get angle between x, y and z = Y/y
    v1 = X[np.newaxis] - Y 
    v2 = Y[:, np.newaxis] - Y
    
    angles = getAngle(v1, v2) # an |Y| x |Y| array
    
    ind = (angles <= omega).prod(axis=-1).astype(bool)
    
    return atoms[ind]

def pairsWithinDistance(struct1, struct2, distance=4.0, skip_hydrogens=False, id_format='biopython', level='R', return_identifier=True, flatten=False, effective_interaction=False, omega=1.571):
    # Get KDTree of structure 1
    if skip_hydrogens:
        alist = []
        for atom in struct1.get_atoms():
            if atom.element == "H":
                continue
            alist.append(atom)
        kdt1 = getAtomKDTree(alist, engine="biopython")
    else:
        kdt1 = struct1.atom_KDTree
    
    # Find all pairs 
    pair_set = set()
    for residue2 in struct2.get_residues():
        for atom2 in residue2:
            if skip_hydrogens and atom2.element == 'H':
                continue
            
            if level == "R":
                residues1 = kdt1.search(atom2.coord, distance, level='R')
                if len(residues1) > 0:
                    pair_set.update([(residue2, residue1) for residue1 in residues1])
            if level == "A":
                atoms1 = kdt1.search(atom2.coord, distance, level='A')
                if len(atoms1) > 0:
                    if effective_interaction:
                        atoms1 = getEffectiveInteractions(atom2, atoms1, omega)
                    pair_set.update([(atom2, atom1) for atom1 in atoms1])
    
    if return_identifier:
        # replace entities with id
        pair_set = [(p[0].get_full_id(), p[1].get_full_id()) for p in pair_set]
    
    if flatten:
        # split pairs into source set and target set
        struct1_items = set()
        struct2_items = set()
        for p in pair_set:
            struct1_items.add(p[1])
            struct2_items.add(p[0])
    
        if return_identifier and id_format == 'dnaprodb':
            if level == "R":
                struct1_items = ["{}.{}.{}".format(_[2], _[3][1], _[3][2]) for _ in  struct1_items]
                struct2_items = ["{}.{}.{}".format(_[2], _[3][1], _[3][2]) for _ in  struct2_items]
            else:
                struct1_items = ["{}.{}.{}.{}".format(_[2], _[3][1], _[3][2], _[4]) for _ in  struct1_items]
                struct2_items = ["{}.{}.{}.{}".format(_[2], _[3][1], _[3][2], _[4]) for _ in  struct2_items]
        else:
            struct1_items = list(struct1_items)
            struct2_items = list(struct2_items)
    
        return struct1_items, struct2_items
    else:
        # return all pairs
        return list(pair_set)
