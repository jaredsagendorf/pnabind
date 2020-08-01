# third party packages
import numpy as np

def getVectorAngle(v1, v2):
    return np.arctan2(np.linalg.norm(np.cross(v1, v2, axis=1), axis=1), (v1 * v2).sum(axis=1))

def getPPFeatures(mesh, edge_index, edge_attr=None):
    
    # vertex indices of each edge
    v1 = edge_index[:,0]
    v2 = edge_index[:,1]
    
    # vector lying along the edge
    e = mesh.vertices[v2] - mesh.vertices[v1]
    
    features = np.stack([
        np.linalg.norm(e, axis=1), # length of edge
        getVectorAngle(mesh.vertex_normals[v1], e), # angle between edge and n1
        getVectorAngle(mesh.vertex_normals[v2], e), # angle between edge and n2
        getVectorAngle(mesh.vertex_normals[v1], mesh.vertex_normals[v2]) # angle between normal 1 and normal 2
    ], axis=1)
    
    if(edge_attr is not None):
        features = np.concatenate([edge_attr, features], axis=1)
    
    return features

def getGeometricEdgeFeatures(mesh):
    mesh = mesh.mesh # access the wrapped mesh
    
    assert np.all(mesh.edges_unique == mesh.face_adjacency_edges)
    
    # get features derived from adjacent faces
    edge_attr = np.stack([
        mesh.face_adjacency_angles,
        mesh.face_adjacency_span,
        ], axis=1
    )
    
    # create undirected edge indices
    edge_index = np.concatenate([
        mesh.edges_unique,
        np.flip(mesh.edges_unique, axis=1)
        ]
    )
    
    edge_attr = np.concatenate([edge_attr, edge_attr])
    
    # get PP Features
    edge_attr = getPPFeatures(mesh, edge_index, edge_attr)
    
    return edge_index, edge_attr
