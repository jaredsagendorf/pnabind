# Raktim Mitra (timkartar7879@gmail.com, raktimmi@usc.edu)
import numpy as np
import trimesh


def smoothMeshLabels(mesh, key, class_label=1, threshold=16.0):
    first_pass(mesh, key, class_label=0)
    first_pass(mesh, key, class_label=1)
    second_pass(mesh, key)
    return mesh

def first_pass(mesh, key, class_label=1, threshold=16.0):
    '''Works for 2 classes only, given as class 0 and class 1'''
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

    map_a = np.argwhere(node_mask).flatten() # index to the original face indices
    component_sizes = np.array([mesh.area_faces[map_a[c]].sum() for c in components])
    #### flip labels where component_size < threshold (total triangle area)
    components_to_flip = np.argwhere(component_sizes < threshold).flatten()
    vertices = []
    for ci in components_to_flip:
        face_idx = map_a[components[ci]]
        vertices.append(mesh.faces[face_idx].flatten())
    if(len(vertices) > 0):
        vertices = np.hstack(vertices)
        vertices = np.unique(vertices)
        mesh.vertex_attributes[key][vertices] = 1 - class_label

def second_pass(mesh, attribute): # remove vertices which has no neighbour from same class
    g = nx.from_edgelist(mesh.edges_unique)
    one_ring = np.array([np.array(list(g[i].keys())) for i in range(len(mesh.vertices))])
    labels = np.sort(np.unique(mesh.vertex_attributes[attribute]))
    for c in labels:
        Vc = np.where(mesh.vertex_attributes[attribute] == c)[0]
        neighbours_per_vertex = one_ring[Vc]
        tochange = np.argwhere(1 - np.array([np.any(mesh.vertex_attributes[attribute][neighbours_per_vertex[i]] == c) for i in range(len(Vc))]))
        tochange = Vc[tochange]
        mesh.vertex_attributes[attribute][tochange] = 1 - c
