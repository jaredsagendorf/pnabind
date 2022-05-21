import warnings

# third party modules
import numpy as np

# geobind modules
from geobind.mesh._zernike import meshDescriptors
from .laplacian_smoothing import laplacianSmoothing
from .map_point_features_to_mesh import mapPointFeaturesToMesh

class MeshDescriptor(object):
    def __init__(self, mesh=None, patches=None, n_max=8, l_max=10, center_patches=True, scale_patches=True):
        # given data members
        self.mesh = mesh
        self.patches = patches
        self.n_max = n_max
        self.l_max = l_max
        self.scale_patches = scale_patches
        self.center_patches = center_patches
        
        # derived data members
        if mesh is not None:
            self.V = mesh.vertices
            self.F = mesh.faces
            self.Nv = len(self.V)
            self.vi_map = np.empty(self.Nv, dtype=np.int64)
        
        # n,l indices
        mask = []
        for n in range(n_max + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    mask.append(l <= l_max)
        self.descriptor_mask = np.array(mask)
    
    def getPatch(self, i):
        # indices of neighboring vertices
        row, col = self.patches[i].nonzero()
        
        # face indices containing vertices in col
        fi = np.unique(self.mesh.vertex_faces[col].flatten())[1:]
        
        # add missing vertices
        vi = np.unique(self.F[fi].flatten())
        self.vi_map[vi] = np.arange(len(vi))
        
        # get new vertices/faces
        faces = self.vi_map[self.F[fi]]
        vertices = self.V[vi]
        
        if self.center_patches:
            vertices = vertices - np.mean(vertices, axis=0)
        
        if self.scale_patches:
            vertices /= np.max(np.sqrt(np.sum(vertices**2, axis=1)))
        
        return vertices, faces
    
    def getPatchDescriptors(self, i=None, patch=None):
        if i is not None:
            # compute patch ourself
            vertices, faces = self.getPatch(i)
        elif patch is not None:
            # use given patch
            vertices, faces = patch.vertices, patch.faces
        
        descriptors, = meshDescriptors(vertices, faces, order=self.n_max, scale_input=False, center_input=False, geometric_moment_invariants=False)
        
        return descriptors[self.descriptor_mask]

def getRadialGeodesicPatches(mesh, radius, add_self_loops=True, to_csr=False):
    try:
        import gdist
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The dependency 'gdist' is required for this functionality!")
    
    V = mesh.vertices.astype(np.float64)
    F = mesh.faces.astype(np.int32)
    
    patches = gdist.local_gdist_matrix(V, F, max_distance=radius)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if add_self_loops:
            patches.setdiag(-1)
        
        if to_csr:
            patches = patches.tocsr()
    
    return patches

def getPatchDescriptors(mesh, radius=15.0, n_max=20, l_max=10, feature_name='zd', sample_ratio=1.0):
    
    patches = getRadialGeodesicPatches(mesh, radius, add_self_loops=True, to_csr=True)
    
    M = MeshDescriptor(mesh, patches, n_max=n_max, l_max=l_max, scale_patches=True, center_patches=True)
    
    if sample_ratio < 1.0:
        # use fps centroids as patch loci
        from torch_cluster import fps
        import torch
        
        idx = fps(torch.tensor(mesh.vertices), batch=None, ratio=sample_ratio, random_start=False)
        idx = idx.numpy()
        descriptors = []
        for i in idx:
            descriptors.append( M.getPatchDescriptors(i) )
        descriptors = np.array(descriptors)
        
        # map to all vertices
        descriptors = mapPointFeaturesToMesh(mesh, mesh.vertices[idx], descriptors, map_to='nearest')
    else:
        # loop over vertices, compute moments for every patch
        descriptors = []
        for i in range(M.Nv):
            descriptors.append( M.getPatchDescriptors(i) )
        descriptors = np.array(descriptors)
    
    # add features to mesh
    feature_names = []
    for i in range(descriptors.shape[1]):
        key = "{}{}".format(feature_name, i)
        mesh.vertex_attributes[key] = descriptors[:,i]
        feature_names.append(key)
    
    return feature_names
