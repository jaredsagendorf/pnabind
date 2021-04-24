import warnings

# third party modules
import gdist
import numpy as np

from geobind.mesh._zernike import meshZernikeMoments
from .laplacian_smoothing import laplacianSmoothing

class Momentor(object):
    def __init__(self, mesh=None, patches=None, order=8, center_patches=True, scale_patches=True):
        # given data members
        self.mesh = mesh
        self.patches = patches
        self.order = order
        self.scale_patches = scale_patches
        self.center_patches = center_patches
        
        # derived data members
        if mesh is not None:
            self.V = mesh.vertices
            self.F = mesh.faces
            self.Nv = len(self.V)
            self.vi_map = np.empty(self.Nv, dtype=np.int64)
    
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
    
    def getPatchMoments(self, i=None, patch=None):
        if i is not None:
            vertices, faces = self.getPatch(i)
        elif patch is not None:
            vertices, faces = patch.vertices, patch.faces
        
        moments = meshZernikeMoments(vertices, faces, order=self.order, scale_input=True)
        
        return np.array(moments)

def getRadialGeodesicPatches(mesh, radius=3.5, add_self_loops=True, to_csr=False):
    
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

def getPatchZernikeMoments(mesh, radius=3.5, order=8, feature_name='zd', smooth=False):
    
    patches = getRadialGeodesicPatches(mesh, radius=radius, add_self_loops=True, to_csr=True)
    
    M = Momentor(mesh, patches, order=order, scale_patches=True, center_patches=True)
    
    # loop over vertices, compute moments for every patch
    moments = []
    for i in range(M.Nv):
        moments.append( M.getPatchMoments(i) )
    moments = np.array(moments)
    
    if smooth:
        moments = laplacianSmoothing(mesh, moments, iterations=1)
    
    feature_names = []
    for i in range(moments.shape[1]):
        key = "{}{}".format(feature_name, i)
        mesh.vertex_attributes[key] = moments[:,i]
        feature_names.append(key)
    
    return feature_names
