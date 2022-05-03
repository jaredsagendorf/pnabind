# third part modules
import numpy as np

# geobind modules
from geobind.mesh import mapPointFeaturesToMesh
from geobind.structure.data import data as D

def mapStructureFeaturesToMesh(mesh, structure, feature_names, residue_ids=None, include_hydrogens=True, impute=True, **kwargs):
    """
        map_to: neighborhood, nearest
    """
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    
    # loop over atoms
    coords = []   # atom coordinates
    radii = []    # atomic radii
    features = [] # atomic features
    F = lambda f: atom.xtra.get(f, 0.0)
    for atom in structure.get_atoms():
        if not include_hydrogens and atom.element == 'H':
            # ignore hydrogen atoms
            continue
        
        if residue_ids:
            # check if we want to include atoms from a particular residue
            aid = atom.get_full_id()
            rid = "{}.{}.{}".format(aid[2], aid[3][1], aid[3][2])
            if not (rid in residue_ids):
                continue
        
        if not impute and not all([fn in atom.xtra for fn in feature_names]):
            # skip if atom missing any features
            print(atom.get_id(), atom.xtra)
            continue
        
        coords.append(atom.coord)
        radii.append(atom.xtra.get("radius", D.getAtomRadius(atom)))
        features.append(list(map(F, feature_names)))
    features = np.array(features)

    return mapPointFeaturesToMesh(mesh, coords, features, offset=radii, **kwargs)
