# third part modules
import numpy as np

# geobind modules
from geobind.mesh import mapPointFeaturesToMesh

def mapStructureFeaturesToMesh(mesh, structure, feature_names, residue_ids=None, hydrogens=False, **kwargs):
    """
        map_to: neighborhood, nearest
    """
    # loop over atoms
    coords = []   # atom coordinates
    radii = []    # atomic radii
    features = [] # atomic features
    F = lambda f: atom.xtra.get(f, 0.0)
    for atom in structure.get_atoms():
        if not hydrogens and atom.element == 'H':
            # ignore hydrogen atoms
            continue
        
        if residue_ids:
            # check if we want to include atoms from a particular residue
            aid = atom.get_full_id()
            rid = "{}.{}.{}".format(aid[2], aid[3][1], aid[3][2])
            if not (rid in residue_ids):
                continue
        
        coords.append(atom.coord)
        radii.append(atom.xtra["radius"])
        features.append(list(map(F, feature_names)))
    features = np.array(features)

    return mapPointFeaturesToMesh(mesh, coords, features, offset=radii, **kwargs)
