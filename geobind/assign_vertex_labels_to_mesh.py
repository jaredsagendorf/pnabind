# builtin modules
import json
import re

# third party modules
import numpy as np
from scipy.spatial import cKDTree

# geobind modules
from geobind.structure.data import data
from geobind.mesh import smoothMeshLabels

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
        resn = residue.get_resname()
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
    
    def testResidue(self, residue):
        """Check if residue is recognized or not"""
        resn = residue.get_resname()
        found = False
        for res_item in self.regexes['residue_regexes']:
            if re.search(res_item['residue_regex'], resn):
                found = True
        
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

def signedVolume(a, b, c, d):
    """Computes the signed volume of a series of tetrahedrons defined by the vertices in 
    a, b c and d. The ouput is an SxT array which gives the signed volume of the tetrahedron defined
    by the line segment 's' and two vertices of the triangle 't'."""
    
    return np.sum((a-d)*np.cross(b-d, c-d), axis=2)

def segmentsIntersectTriangles(s, t):
    """For each line segment in 's', this function computes whether it intersects any of the triangles
    given in 't'."""
    # compute the normals to each triangle
    normals = np.cross(t[2]-t[0], t[2]-t[1])
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    
    # get sign of each segment endpoint, if the sign changes then we know this segment crosses the
    # plane which contains a triangle. If the value is zero the endpoint of the segment lies on the 
    # plane.
    # s[i][:, np.newaxis] - t[j] -> S x T x 3 array
    sign1 = np.sign(np.sum(normals*(s[0][:, np.newaxis] - t[2]), axis=2)) # S x T
    sign2 = np.sign(np.sum(normals*(s[1][:, np.newaxis] - t[2]), axis=2)) # S x T
    
    # determine segments which cross the plane of a triangle. 1 if the sign of the end points of s is 
    # different AND one of end points of s is not a vertex of t
    cross = (sign1 != sign2)*(sign1 != 0)*(sign2 != 0) # S x T 
    
    # get signed volumes
    v1 = np.sign(signedVolume(t[0], t[1], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v2 = np.sign(signedVolume(t[1], t[2], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v3 = np.sign(signedVolume(t[2], t[0], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    
    same_volume = np.logical_and((v1 == v2), (v2 == v3)) # 1 if s and t have same sign in v1, v2 and v3
    
    return np.nonzero(np.logical_not(np.sum(cross*same_volume, axis=1)))[0]

def assignMeshLabelsFromStructure(structure, mesh, atom_mapper,
        distance_cutoff=4.0,
        hydrogens=True,
        check_for_intersection=True,
        smooth=False,
        mask=False,
        mask_cutoff=5.0
    ):
    
    nc = atom_mapper.nc
    Y = np.zeros((len(mesh.vertices), nc)) # V x C one hot encoding, 0 being the default class
    Y[:,0] += 1e-5 # add small value to default class to avoid possible ties  
    for atom in structure.get_atoms():
        # check if we include hydrogens
        if not hydrogens and atom.element == 'H':
            continue
        
        # assign a class to this atom
        residue = atom.get_parent()
        c = atom_mapper(residue, atom)
        
        # get nearest vertices
        v, d = mesh.verticesInBall(atom.coord, distance_cutoff)
        
        if(len(v) > 0):
            w = np.clip(1/(d+1e-5), 0.0, 2.0)
            
            if(check_for_intersection):
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
        Y = smoothMeshLabels(mesh.edges, Y, nc, faces=mesh.faces, area_faces=mesh.area_faces, threshold=50.0)
    
    if mask :
        # mask the boundaries of any binding region
        Yn = np.argwhere(Y == 0).flatten()
        Yb = np.argwhere(Y != 0).flatten()
        if len(Yb) > 0:
            Kn = cKDTree(mesh.vertices[Yn])
            query = Kn.query_ball_point(mesh.vertices[Yb], mask_cutoff) # all non-bs vertices within `mask_cutoff` of bs vertices
            ind = [i for j in query for i in j] # need to flatten this list of lists
            Y[Yn[ind]] = -1
    
    return Y

def assignMeshLabelsFromList(structure, mesh, residue_ids, distance_cutoff=2.5, smooth=False):
    # loop over residues
    pass
