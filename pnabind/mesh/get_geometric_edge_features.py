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
    
    if edge_attr is not None:
        features = np.concatenate([edge_attr, features], axis=1)
    
    return features

def getGeometricEdgeFeatures(mesh, directed_edges=True, pp_features=True, triangle_features=True, n_components=0, normalize=True):
    assert np.all(mesh.edges_unique == mesh.face_adjacency_edges)
    
    edge_attr = []
    if triangle_features:
        # get undirected edge features derived from adjacent triangle faces
        edge_attr.append(mesh.face_adjacency_angles)
        edge_attr.append(mesh.face_adjacency_span)
        edge_attr.append(mesh.area_faces[mesh.face_adjacency].sum(axis=1)/3)
        
        # edges of one-ring
        vec_ua = mesh.vertices[mesh.face_adjacency_unshared[:,0]] - mesh.vertices[mesh.edges_unique[:,0]]
        vec_ub = mesh.vertices[mesh.face_adjacency_unshared[:,1]] - mesh.vertices[mesh.edges_unique[:,0]]
        vec_va = mesh.vertices[mesh.face_adjacency_unshared[:,0]] - mesh.vertices[mesh.edges_unique[:,1]]
        vec_vb = mesh.vertices[mesh.face_adjacency_unshared[:,1]] - mesh.vertices[mesh.edges_unique[:,1]]
        vec_uv = mesh.vertices[mesh.edges_unique[:,1]] - mesh.vertices[mesh.edges_unique[:,0]]
        
        # sum of the adjacent interior angles
        edge_attr.append(getVectorAngle(vec_ua, vec_uv) + getVectorAngle(vec_ub, vec_uv) + getVectorAngle(vec_va, vec_uv) + getVectorAngle(vec_vb, vec_uv))
        
        # sum of the opposite interior angles
        edge_attr.append(getVectorAngle(vec_ua, vec_va) + getVectorAngle(vec_ub, vec_vb))
        
        # combine edge features
        edge_attr = np.stack(edge_attr, axis=1)
    
    if directed_edges:
        # create undirected edges
        edge_index = np.concatenate([
            mesh.edges_unique,
            np.flip(mesh.edges_unique, axis=1)
            ]
        )
        edge_attr = np.concatenate([edge_attr, edge_attr])
    
        # get PP Features
        if pp_features:
            edge_attr = getPPFeatures(mesh, edge_index, edge_attr)
    
    if normalize:
        from sklearn.preprocessing import scale
        edge_attr = scale(edge_attr)
    
    if n_components > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, copy=True, svd_solver="arpack")
        pca.fit(edge_attr)
        edge_attr = pca.transform(edge_attr)
    
    return edge_index, edge_attr
