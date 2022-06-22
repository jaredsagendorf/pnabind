# builtin modules
import json
import re
import logging

# third party modules
import numpy as np
try:
    from scipy.spatial import cKDTree
except BaseException as E:
    from .exceptions import ExceptionModule
    cKDTree = ExceptionModule(E)

# geobind modules
from geobind.structure.data import data
from geobind.mesh import smoothMeshLabels
from geobind.utils import segmentsIntersectTriangles
from .map_structure_features_to_mesh import mapStructureFeaturesToMesh

class AtomToClassMapper(object):
    def __init__(self, ligand_info, default=0, name="LIGANDS"):
        
        if isinstance(ligand_info, str):
            # check if it's a pre-built label set, otherwise assume it's a file name
            if ligand_info in data.label_sets:
                self.regexes = data.label_sets[ligand_info]
            else:
                with open(ligand_info) as FH:
                    self.regexes = json.load(FH)
        elif isinstance(ligand_info, list):
            self.regexes = {
                "residue_regexes": [],
                "nc": len(ligand_info)+1,
                "default": default,
                "classes": {
                    default: "default",
                }
            }
            for i in range(len(ligand_info)):
                ligand = ligand_info[i]
                self.regexes["classes"][i+1] = ligand
                self.regexes["residue_regexes"].append(
                    {
                        "residue_regex": ligand,
                        "group": ligand,
                        "atom_regexes": [
                            {
                                "atom_regex": ".*",
                                "class": i+1
                            }
                        ]
                    }
                )
        else:
            self.regexes = ligand_info
        self.default = self.regexes['default']
        self.nc = self.regexes['nc']
        self.classes = self.regexes['classes']
        if "name" in self.regexes:
            self.name = self.regexes["name"]
        else:
            self.name = name
        
        assert isinstance(self.default, int)
    
    def __call__(self, residue, atom=None, hydrogens=True):
        resn = residue.get_resname().strip()
        atmn = atom.name.strip()
        if atom.element != 'H':
            # a non-hydrogen atom
            for res_item in self.regexes['residue_regexes']:
                if re.search(res_item['residue_regex'], resn):
                    # found matching residue group, iterate over atom regexes
                    for atm_item in res_item['atom_regexes']:
                        if re.search(atm_item['atom_regex'], atmn):
                            # found matching atom group
                            return atm_item['class']
        else:
            # a hydrogen atom, use class of parent heavy atom
            if hydrogens:
                parent_atom = AtomToClassMapper.getParentAtom(residue, atom)
                return self.__call__(residue, parent_atom)
        
        # no match found, return default class
        return self.default
    
    def testResidue(self, residue, allow_modified=True):
        """Check if residue is recognized or not"""
        def _test(self, name):
            found = False
            for res_item in self.regexes['residue_regexes']:
                if re.search(res_item['residue_regex'], resn):
                    found = True
                    break
            return found
        
        resn = residue.get_resname().strip()
        found = _test(self, resn)
        
        if (not found) and allow_modified:
            # check chemical components parent
            if resn in data.chem_components:
                parn = data.chem_components[resn]['_chem_comp.mon_nstd_parent_comp_id'].strip()
                found = _test(self, parn)
        
        return found
    
    @classmethod
    def getParentAtom(cls, residue, atom):
        children = residue.get_atoms()
        min_dist = 9999999
        parent = None
        for child in children:
            if child == atom:
                continue
            elif child.element == 'H':
                continue
            else:
                dist = atom-child
                if dist < min_dist:
                    min_dist = dist
                    parent = child
        return parent

def assignMeshLabelsFromStructure(structure, mesh, atom_mapper,
        distance_cutoff=4.0,
        include_hydrogens=True,
        check_for_intersection=True,
        smooth=False,
        smoothing_threshold=50.0,
        no_smooth=None,
        mask=False,
        mask_cutoff=4.0
    ):
    
    nc = atom_mapper.nc
    Y = np.zeros((len(mesh.vertices), nc)) # V x C one hot encoding, 0 being the default class
    Y[:,0] += 1e-5 # add small value to default class to avoid possible ties  
    for atom in structure.get_atoms():
        # check if we include hydrogens
        if not include_hydrogens and atom.element == 'H':
            continue
        
        # assign a class to this atom
        residue = atom.get_parent()
        c = atom_mapper(residue, atom)
        
        # get nearest vertices
        v, d = mesh.verticesInBall(atom.coord, distance_cutoff)
        
        if len(v) > 0:
            w = np.clip(1/(d+1e-5), 0.0, 2.0)
            
            if check_for_intersection:
                # check if atom-vertex segments intersect the mesh
                t = mesh.facesInBall(atom.coord, distance_cutoff)
                ind = segmentsIntersectTriangles(
                    (np.tile(atom.coord, (len(v),1)), mesh.vertices[v]),
                    (mesh.vertices[t[:,0]], mesh.vertices[t[:,1]], mesh.vertices[t[:,2]])
                )
                v = v[ind]
                w = w[ind]
            
            # add weights to labels
            Y[v, c] += w
    Y = np.argmax(Y, axis=1)
    
    if smooth:
        # smooth labels using a smoothing scheme
        Y = smoothMeshLabels(mesh.edges, Y, nc, 
            faces=mesh.faces,
            area_faces=mesh.area_faces,
            threshold=smoothing_threshold,
            ignore_class=no_smooth
        )
    
    if mask :
        label_mask = maskClassBoundary(mesh.vertices, Y, mask_cutoff=mask_cutoff, masked_class=0)
        Y[~label_mask] = -1
    
    return Y

def assignMeshLabelsFromList(model, mesh, residues,
        cl=1,
        nc=2,
        hydrogens=True,
        smooth=False,
        smoothing_threshold=50.0,
        no_smooth=None,
        mask=False,
        mask_cutoff=4.0,
        feature_name="bs",
        **kwargs
    ):
    
    # loop over residues
    key = feature_name+str(cl)
    for residue in residues:
        for atom in residue:
            atom.xtra[key] = cl
    
    # map to vertices
    Y = mapStructureFeaturesToMesh(mesh, model, [key], include_hydrogens=hydrogens, distance_cutoff=1.5, **kwargs)
    Y = np.round(Y.reshape(-1)).astype(int)
    
    if smooth:
        # smooth labels using a smoothing scheme
        Y = smoothMeshLabels(mesh.edges, Y, nc, 
            faces=mesh.faces,
            area_faces=mesh.area_faces,
            threshold=smoothing_threshold,
            ignore_class=no_smooth
        )
    
    if mask :
        label_mask = maskClassBoundary(mesh.vertices, Y, mask_cutoff=mask_cutoff, masked_class=0)
        Y[~label_mask] = -1
    
    return Y

def maskClassBoundary(vertices, Y, mask_cutoff=4.0, masked_class=0, return_type=bool):
    # mask the boundaries of any binding region
    m_ind = np.argwhere(Y == masked_class).flatten() # indices of class to be masked
    n_ind = np.argwhere(Y != masked_class).flatten() # indices of other classes
    mask = np.ones_like(Y)
    if len(n_ind) > 0 and len(m_ind) > 0:
        Km = cKDTree(vertices[m_ind])
        query = Km.query_ball_point(vertices[n_ind], mask_cutoff) # all 'm' vertices within `mask_cutoff` of 'n' vertices
        ind = [i for j in query for i in j] # need to flatten this list of lists
        mask[m_ind[ind]] = 0 # set mask to zero for 'm' vertices
    
    return mask.astype(return_type)
