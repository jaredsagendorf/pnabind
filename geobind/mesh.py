# Jared Sagendorf

"""A simple set of methods and classes for dealing with and performing basic operations on 
triangular meshes."""
import os
import numpy as np
import networkx as nx
from matplotlib import cm
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

from .mesh_io import writeOFF, readOFF

class BoundingBox(object):
    """Simple class for constructing the eight corners defining the rectangular bounding box
    of a set of points."""
    def __init__(self, points):
        self.points = points
        xmin, ymin, zmin = points[:,0].min(), points[:,1].min(), points[:,2].min()
        xmax, ymax, zmax = points[:,0].max(), points[:,1].max(), points[:,2].max()
        self.corners = np.array([
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
            [xmin, ymax, zmin],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmax, ymax, zmin]
        ])
        self.edges = [
            (self.corners[0], self.corners[1]),
            (self.corners[1], self.corners[2]),
            (self.corners[2], self.corners[3]),
            (self.corners[3], self.corners[0]),
            
            (self.corners[4], self.corners[5]),
            (self.corners[5], self.corners[6]),
            (self.corners[6], self.corners[7]),
            (self.corners[7], self.corners[4]),
            
            (self.corners[0], self.corners[4]),
            (self.corners[1], self.corners[5]),
            (self.corners[2], self.corners[6]),
            (self.corners[3], self.corners[7]),
        ]
        self.__max_length = None
        self.__min_length = None
        self.__aspect_ratio = None
    
    @property
    def max_length(self):
        if(self.__max_length is None):
            # iterate over edges to find the longest
            self.__max_length = float('-inf')
            for e in self.edges:
                self.__max_length = max(np.linalg.norm(e[1]-e[0]), self.__max_length)
        return self.__max_length
    @max_length.setter
    def max_length(self, l):
        if(l < 0):
            raise ValueError("Length must be non-zero.")
        else:
            self.__max_length = l
    
    @property
    def min_length(self):
        if(self.__min_length is None):
            # iterate over edges to find the shortest
            self.__min_length = float('inf')
            for e in self.edges:
                self.__min_length = min(np.linalg.norm(e[1]-e[0]), self.__min_length)
        return self.__min_length
    @min_length.setter
    def min_length(self, l):
        if(l < 0):
            raise ValueError("Length must be non-zero.")
        else:
            self.__min_length = l
    
    @property
    def aspect_ratio(self):
        if(self.__aspect_ratio is None):
            self.__aspect_ratio = self.max_length/self.min_length
        return self.__aspect_ratio
    @aspect_ratio.setter
    def aspect_ratio(self, ar):
        if(ar < 1.0):
            raise ValueError("Ratio must be greater than or equal to 1.0.")
        else:
            self.__aspect_ratio = ar

class Mesh(object):
    """A simple class for peforming some basic operations on a triangular mesh"""
    def __init__(self, handle=None, V=None, F=None, N=None, name="mesh", clean=True):
        if(handle is not None):
            if(isinstance(handle, str)):
                ext = handle.split('.')[-1]
                if(ext.lower() == 'off'):
                    self.V, self.F = readOFF(handle)
                elif(ext.lower() == 'ply'):
                    self.V, self.F = readPLY(handle)
                else:
                    raise ValueError("Unknown file extention for mesh input: {}.".format(ext))
            else:
                raise TypeError("Mesh file handle must be a string")
        elif((V is not None) and (F is not None)):
            # Check that these are valid sizes/types
            if(V.shape[1] == 3 and V.ndim == 2):
                self.V = V
            else:
                raise ValueError("Vertices must be an Vx3 array")
            if(F.shape[1] == 3 and F.ndim == 2):
                self.F = F
            else:
                raise ValueError("Faces must be an Fx3 array")
        else:
            raise Exception("Insuffucient data given to construct a mesh object")
        
        # Properties of the mesh, some of which are determined via setter/getter functions
        self.name = name
        self.nV = self.V.shape[0] # number of vertices in the mesh
        self.nF = self.F.shape[0] # number of faces in the mesh
        self.nNv = None # number of vertex normals in the mesh
        self.nNf = None # number of face normals in the mesh
        self.__bbox = None # bounding box
        self.__area = None # total mesh area
        self.__volume = None # total mesh volume
        self.__triangle_areas = None # #F X 1 array of triange face areas
        self.__KDTree = None # KDTree for NN lookup
        self.__Nv = None # vertex normals
        self.__Nf = None # face normals
        self.__VG = None # vertex connectivity graph
        self.__VA = None # vertex adjacency matrix
        self.__FG = None # face connectivity graph
        self.__V2F = None # maps a vertex index to all faces it belongs to
        
        # Check if we're given normal vectors
        if(N is not None):
            if(N.shape[1] == 3 and N.ndim == 2 and N.shape == self.V.shape):
                self.__Nv = N
                self.nNv = self.__Nv.shape[0]
            else:
                raise TypeError("Normals must be an Nx3 array equal in shape to the vertices array.")
        
        if(clean):
            # remove any spurious vertices
            self.clean()
    
    @property
    def triangle_areas(self):
        if(self.__triangle_areas is None):
            # compute area using the formula A = |BA X CA|/2 for a triangle defined by
            # vectors A, B and C
            cross = np.cross(
                self.V[self.F[:,1]] - self.V[self.F[:,0]],
                self.V[self.F[:,2]] - self.V[self.F[:,0]]
            )
            self.__triangle_areas = np.linalg.norm(cross, axis=1)/2.0
        return self.__triangle_areas
    @triangle_areas.setter
    def triangle_areas(self, ta):
        if(ta.shape == self.F.shape):
            self.__triangle_areas = ta
        else:
            raise ValueError("Triangle area array does not match faces shape.")
    
    @property
    def area(self):
        if(self.__area is None):
            self.__area = self.triangle_areas.sum()
        return self.__area
    @area.setter
    def area(self, a):
        if(a < 0):
            raise ValueError("Mesh area must be non-zero.")
        else:
            self.__area = a
    
    @property
    def volume(self):
        if(self.__volume is None):
            # compute volume using the formula V = (A X B)*C/6 for a tetrahedron defined by
            # vectors A, B and C and the origin
            cross = np.cross(self.V[self.F[:,0]], self.V[self.F[:,1]])
            signed_volumes = np.einsum('ij,ij->i', cross, self.V[self.F[:,2]])
            self.__volume = np.abs(signed_volumes.sum())/6.0
        
        return self.__volume
    @volume.setter
    def volume(self, v):
        if(v < 0):
            raise ValueError("Mesh volume must be non-zero.")
        else:
            self.__area = v
    
    @property
    def bbox(self):
        if(self.__bbox is None):
            # Construct a bounding box object
            self.__bbox = BoundingBox(self.V)
        return self.__bbox
    @bbox.setter
    def bbox(self, b):
        if(isinstance(b, BoundingBox)):
            self.__bbox = b
        else:
            raise ValueError("Not a valid bounding box instance!")
    
    @property
    def KDTree(self):
        if(self.__KDTree is None):
            self.__KDTree = cKDTree(self.V)
        return self.__KDTree
    @KDTree.setter
    def KDTree(self, k):
        self.__KDTree = k
    
    # vertex graph
    @property
    def VG(self):
        if(self.__VG is None):
            # Generate connectivity graph
            G = nx.Graph()
            for f in range(self.nF):
                i = self.F[f][0]
                j = self.F[f][1]
                k = self.F[f][2]
                G.add_edge(i,j)
                G.add_edge(i,k)
                G.add_edge(j,k)
            
            self.__VG = G
        
        return self.__VG
    @VG.setter
    def VG(self, G):
        self.__VG = G
    
    # vertex adjacency matrix
    @property
    def VA(self):
        if(self.__VA is None):
            self.__VA = coo_matrix(nx.adjacency_matrix(self.VG, nodelist=[i for i in range(self.nV)]))
        return self.__VA
    
    # face graph
    @property
    def FG(self):
        return self.__FG
    
    # vertex normals
    @property
    def Nv(self):
        return self.__Nv
    @Nv.setter
    def Nv(self, normals):
        if(isinstance(normals, str)):
            # load the normals from file
            normals = np.loadtxt(normals, dtype=np.float32)
        
        # check size of normals
        if(not (normals.shape[1] == 3 and normals.ndim == 2)):
            raise ValueError("Normals must be an Nx3 array")
        if(normals.shape[0] != self.nV):
            raise ValueError("Size of normals array ({}) does not match size of vertex array ({})!".format(normals.shape[0], self.nV))
        self.__Nv = normals
        self.nNv = normals.shape[0]
    
    @property
    def V2F(self):
        """V2F stores the list of face indices that each vertex belongs to."""
        if(self.__V2F is None):
            self.__V2F = [[] for _ in range(self.nV)]
            for i in range(len(self.F)):
                self.__V2F[self.F[i][0]].append(i)
                self.__V2F[self.F[i][1]].append(i)
                self.__V2F[self.F[i][2]].append(i)
        return self.__V2F
    
    def clean(self, smooth=False):
        """Remove disconnected subcomponents"""
        components = list(nx.connected_components(self.VG))
        if(len(components) > 1):
            # need to remove all components but one
            max_order = 0
            max_G = None
            for c in components:
                if(len(c) > max_order):
                    max_order = len(c)
                    max_G = c
            
            # decide which vertices and faces to keep
            vi_list = list(max_G)
            fi_list = set()
            for i in range(len(vi_list)):
                fi_list.update(self.V2F[vi_list[i]])
            fi_list = list(fi_list)
            
            self.V = self.V[vi_list]
            self.F = self.F[fi_list]
            
            # rename vertices in faces
            vim = np.zeros(self.nV)
            vim[vi_list] = np.arange(len(vi_list))
            self.F[:,0] = vim[self.F[:,0]]
            self.F[:,1] = vim[self.F[:,1]]
            self.F[:,2] = vim[self.F[:,2]]
            
            # reset everything that may be affected
            self.nV = self.V.shape[0] # number of vertices in the mesh
            self.nF = self.F.shape[0] # number of faces in the mesh
            self.nNv = None # number of vertex normals in the mesh
            self.nNf = None # number of face normals in the mesh
            self.__bbox = None # bounding box
            self.__area = None # total mesh area
            self.__volume = None # total mesh volume
            self.__triangle_areas = None # F X 1 array of triange face areas
            self.__KDTree = None # KDTree for NN lookup
            self.__Nv = None # vertex normals
            self.__Nf = None # face normals
            self.__VG = None # vertex connectivity graph
            self.__VA = None # vertex adjacency matrix
            self.__FG = None # face connectivity graph
            self.__V2F = None # maps a vertex index to all faces it belongs to
            
            if(self.Nv):
                self.Nv = self.Nv[vi_list]
        
        # perform laplacian smoothing
        if(smooth):
            V = np.zeros(self.nV, 3)
            for node in self.VG:
                pass
    
    def findNearestVertex(self, x):
        """Returns the distance and vertex index which is nearest the given point x"""
        d, i = self.KDTree.query(x)
        return i, d
    
    def findVerticesInBall(self, x, r):
        """Returns all verices within a radius 'r' of a point 'x'"""
        indices = self.KDTree.query_ball_point(x, r)
        indices = np.array(indices, dtype=np.int32)
        distances = np.linalg.norm(self.V[indices] - x, axis=1)
        
        return indices, distances
    
    def findTrianglesInBall(self, x, r):
        """Returns all triangles that contain at least one vertex within radius 'r' of point 'x'"""
        indices = self.KDTree.query_ball_point(x, r)
        ti = set()
        for i in indices:
            ti.update(self.V2F[i])
        return self.F[list(ti)]
        
    def findNeighbors(self, v, k=1, nlist=None, vo=None):
        """Returns the k-neighbors of a given vertex"""
        if(nlist is None):
            nlist = set()
        if(k == 0):
            return
        if(vo is None):
            vo = v
        for n in self.VG.neighbors(v):
            if(n == vo):
                continue
            nlist.add(n)
            self.findNeighbors(n, k=k-1, nlist=nlist, vo=vo)
        return nlist
    
    def save(self, directory=".", file_name=None, file_format="off", overwrite=False):
        """Writes the mesh to file"""
        if(file_name is None):
            file_name = self.name
        
        path = os.path.join(directory, file_name)
        exists = os.path.exists(path) # check if file already exists
        if(exists and not overwrite):
            # just return the existing path and do not overwrite
            return path
        
        # decide which file format to write to
        if(file_format == "off"):
            writeOFF(path, self.V, self.F)
            return path+".off"
        elif(file_format == "ply"):
            writePLY(path, self.V, self.F)
            return path+".ply"
        elif(file_format == "npy"):
            np.save(os.path.join(path, "_vertices.npy"), self.V, allow_pickle=False)
            np.save(os.path.join(path, "_faces.npy"), self.F, allow_pickle=False)
        else:
            raise ValueError("Unrecognized mesh output format: {}".format(file_format))
