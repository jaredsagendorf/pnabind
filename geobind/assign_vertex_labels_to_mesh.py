# builtin modules
import json
import re

# third party modules
import numpy as np
from scipy.spatial import cKDTree

# geobind modules
from geobind.structure.data import data

class AtomToClassMapper(object):
    def __init__(self, regexes, structure=None):
        
        if isinstance(regexes, str):
            # check if it's a pre-built label set, otherwise assume it's a file name
            if regexes in data.label_sets:
                self.regexes = data.label_sets[regexes]
            else:
                with open(regexes_file) as FH:
                    self.regexes = json.load(FH)
        else:
            self.regexes = regexes
        self.default = self.regexes['default']
        self.nc = self.regexes['nc']
        self.structure = structure
        
        assert isinstance(self.default, int)
    
    def __call__(self, residue, atom):
        
        resn = residue.get_resname()
        atmn = atom.name.strip()
        
        if atom.element != 'H':
            # a non-hydrogen atom
            for res_item in self.regexes['regexes']:
                if re.search(res_item['re'], resn):
                    # found matching residue group, iterate over atom regexes
                    for atm_item in res_item['atom_regexes']:
                        if re.search(atm_item['re'], atmn):
                            # found matching atom group
                            return atm_item['class']
        else:
            # a hydrogen atom, use class of parent heavy atom
            if self.structure is not None:
                parent_atom = self.structure.getNearestNeighbor(atom, hydrogens=False)
                if(parent_atom.element == 'H'):
                    print(residue, atom, parent_atom)
                    exit(0)
                return self.__call__(parent_atom.get_parent(), parent_atom)
        
        # no match found, return default class
        return self.default

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

def smoothLabels(mesh, key, class_label=1, threshold=16.0):
    # Laplacian smoothing
    #Yh1 = oneHotEncode(Y, nc)
    #Yh2 = 2*Yh1 # count each vertex as two votes
    #for i in range(mesh.nV):
        #neighbors = mesh.findNeighbors(i)
        #for n in neighbors:
            #Yh2[i] += Yh1[n]
    #Y = np.argmax(Yh2, axis=1)
    
    #### generate face labels taking majority vote of vertex labels
    y_face = ((mesh.vertex_attributes[key][mesh.faces] == class_label).sum(axis=1) >= 2).astype(np.int32)
    Nf = y_face.shape[0]
    node_mask = (y_face == 1)
    Nc = node_mask.sum()
    
    #### create a graph with <class> faces as vertices and connect two vertices if the faces share an edge. 
    E_face = mesh.face_adjacency
    edge_mask = node_mask[E_face[:,0]]*node_mask[E_face[:,1]] # edges where both nodes are a <class> face
    
    # create an index from 0..Nf-1 to 0..Nc-1 when we apply the node mask or edge mask
    map_c = np.empty(Nf, dtype=np.int32)
    map_c[node_mask] = np.arange(Nc)
    
    # map the <class> edges to be within range of 0..Nc-1 and make undirected
    c_edges = map_c[E_face[edge_mask]]
    e1, e2 = c_edges[:, 0], c_edges[:, 1]
    e1, e2 = np.hstack((e1, e2)), np.hstack((e2, e1))
    c_edges = np.stack((e1, e2), axis=1)
    
    # get <class> nodes from 0..Nc-1
    c_nodes = map_c[node_mask]
    
    #### find all connected components in the <class> faces graph
    components = trimesh.graph.connected_components(c_edges, min_len=0, nodes=c_nodes, engine='scipy')
    
    map_a = np.argwhere(node_mask).flatten() # index to the the original face indices
    component_sizes = np.array([mesh.area_faces[map_a[c]].sum() for c in components])
    
    #### flip labels where component_size < threshold (total triangle area)
    components_to_flip = np.argwhere(component_sizes < threshold).flatten()
    vertices = []
    for ci in components_to_flip:
        face_idx = map_a[components[ci]]
        vertices.append(mesh.faces[face_idx].flatten())
    
    if(len(vertices) > 0):
        vertices = np.hstack(vertices)
        mesh.vertex_attributes[key][vertices] = 1 - class_label

def assignMeshLabelsFromStructure(structure, mesh, atom_to_class,
        distance_cutoff=4.0,
        hydrogens=True,
        check_for_intersection=True,
        smooth=False,
        mask=False,
        mask_cutoff=5.0
    ):
    
    atom_mapper = AtomToClassMapper(atom_to_class, structure)
    nc = atom_mapper.nc
    
    Y = np.zeros((len(mesh.vertices), nc)) # V x C one hot encoding, 0 being the default class
    Y[:,0] += 1e-5 # add small value to default class to avoid possible ties  
    for atom in structure.get_atoms():
        # check if we include hydrogens
        if(not hydrogens and atom.element == 'H'):
            continue
        #aname = atom.name.strip()
        residue = atom.get_parent()#.get_resname().strip()
        
        # assign a class to this atom
        #if(nc == 2):
        #    c = 1
        #else:
        #    c = atom_to_class[(rname, aname)]
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
    
    if(smooth):
        # smooth labels using a smoothing scheme
        Y = smoothLabels(mesh, Y, 0)
        Y = smoothLabels(mesh, Y, 1)
    
    if(mask):
        # mask the boundaries of any binding region
        Yn = np.argwhere(Y == 0).flatten()
        Yb = np.argwhere(Y != 0).flatten()
        Kn = cKDTree(mesh.vertices[Yn])
        ind = [i for j in Kn.query_ball_point(mesh.vertices[Yb], mask_cutoff) for i in j]
        Y[Yn[ind]] = -1
    
    return Y

def assignMeshLabelsFromList(structure, mesh, residue_ids, distance_cutoff=2.5, smooth=False):
    # loop over residues
    pass
