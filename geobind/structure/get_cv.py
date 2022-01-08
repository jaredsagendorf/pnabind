# third party packages
import numpy as np

# geobind packages
from .get_residue_id import getResidueID
from .get_atom_kdtree import getAtomKDTree
from .data import data

def getCV(structure, radius, residue_ids=None, ns=None, feature_name="cv", hydrogens=False, bonds=None):
    """ get CV values for every residue in residues"""
    
    if residue_ids is None:
        # use all residues in the structure
        residue_ids = []
        for chain in structure.get_chains():
            for residue in chain:
                residue_ids.append(getResidueID(residue))
    
    if ns is None:
        # create a KDtree for structure atoms
        ns = structure.atom_KDTree
        
    for resID in residue_ids:
        cid, num, ins = resID.split('.')
        residue = structure.get_residue((' ', int(num), ins), cid)
        for atom in residue:
            if not hydrogens and atom.element == 'H':
                continue
            vector = np.zeros(3)
            neighbors = ns.search(atom.get_coord(), radius, level='A')
            for n in neighbors:
                if n == atom:
                    continue
                if not hydrogens and n.element == 'H':
                    continue
                vector += (atom.get_coord() - n.get_coord())/(np.linalg.norm(atom.get_coord() - n.get_coord()) + 1e-5)
            atom.xtra[feature_name] = (1 - np.linalg.norm(vector))/(len(neighbors) - 1 + 1e-5)
    
    # use parent atom as hydrogen CV value if we exluded them
    if not hydrogens:
        if bonds is None:
            # use default bond data
            bonds = data.covalent_bond_data
        
        for resID in residue_ids:
            cid, num, ins = resID.split('.')
            residue = structure.get_residue((' ', int(num), ins), cid)
            resn = residue.get_resname().strip()
            for atom in residue:
                if atom.element != 'H':
                    continue
                aname = atom.get_name().strip()
                parent_atom = bonds[resn][aname]['bonded_atoms'][0]
                if parent_atom in residue:
                    atom.xtra[feature_name] = residue[parent_atom].xtra[feature_name]
    
    return [feature_name]
