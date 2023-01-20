import numpy as np

def getFaceArea(V, F):
    area_faces = np.linalg.norm(np.cross(V[F[:,0]]-V[F[:,1]], V[F[:,0]]-V[F[:,2]]), axis=-1)/2
    
    return area_faces

def getVertexArea(vertices, faces):
    
    # compute face areas
    area_faces = getFaceArea(vertices, faces)
    
    # compute vertex areas
    area_vertices = np.zeros(len(vertices))
    np.add.at(area_vertices, faces[:, 0], area_faces/3)
    np.add.at(area_vertices, faces[:, 1], area_faces/3)
    np.add.at(area_vertices, faces[:, 2], area_faces/3)
    
    return area_vertices
