import numpy as np

# geobind modules
from geobind.structure import StructureData
from geobind.structure import getAtomKDTree
from geobind.structure.data import data
from geobind.structure import mapPointFeaturesToStructure

def vertexLabelsToResidueLabels(atoms, mesh, Y, nc=2, kdt=None, id_format='biopython', null_class=0):    
    if isinstance(atoms, StructureData):
        atoms = atoms.get_atoms()
    
    if kdt is None:
        kdt = getAtomKDTree(atoms)
    
    # get vertex areas
    areas = np.zeros_like(Y, dtype=np.float32)
    np.add.at(areas, mesh.faces[:, 0], mesh.area_faces/3)
    np.add.at(areas, mesh.faces[:, 1], mesh.area_faces/3)
    np.add.at(areas, mesh.faces[:, 2], mesh.area_faces/3)
    
    for c in range(nc):
        mask = (Y == c)
        mapPointFeaturesToStructure(mesh.vertices, atoms, areas*mask, 'area_{}'.format(c), kdtree=kdt)
    
    residue_dict = {}
    # aggregate over atom areas
    for atom in atoms:
        residue = atom.get_parent()
        residue_id = residue.get_full_id()
        if id_format == 'dnaprodb':
            residue_id = '{}.{}.{}'.format(residue_id[2], residue_id[3][1], residue_id[3][2])
        
        if residue_id not in residue_dict:
            residue_dict[residue_id] = {
                'residue_name': residue.get_resname(),
                'class_areas': np.zeros(nc)
            }
        
        for c in range(nc):
            key = 'area_{}'.format(c)
            if key in atom.xtra:
                residue_dict[residue_id]['class_areas'][c] += atom.xtra[key]
    
    # determine residue class
    for residue_id in residue_dict:
        ci = np.argmax(residue_dict[residue_id]['class_areas'])
        resn = residue_dict[residue_id]['residue_name']
        
        if residue_dict[residue_id]['class_areas'][ci] >= data.buried_sesa_cutoffs[resn]:
            residue_dict[residue_id]['label'] = ci
        else:
            residue_dict[residue_id]['label'] = null_class
    
    return residue_dict
