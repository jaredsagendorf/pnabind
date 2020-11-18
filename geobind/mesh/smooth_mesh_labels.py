import numpy as np
import trimesh

def smoothMeshLabels(edges, labels, num_classes, threshold=16.0, faces=None, area_faces=None):
    nodes = np.arange(len(labels))
    # compute an area for each node
    if faces is not None and area_faces is not None:
        # use face areas to assign node weights
        node_areas = np.zeros(len(nodes))
        np.add.at(node_areas, faces[:, 0], area_faces/3)
        np.add.at(node_areas, faces[:, 1], area_faces/3)
        np.add.at(node_areas, faces[:, 2], area_faces/3)
    else:
        # equal weighting for every node
        node_areas = np.ones(len(nodes))
    
    # determine connected components based on class and adjacency and assign each cluster to a 
    # class label
    edge_mask = (labels[edges[:,0]] == labels[edges[:,1]]) # edges where both nodes agree on class 
    clusters = trimesh.graph.connected_components(edges[edge_mask], nodes=nodes) # clusters of connected vertices that have same class
    num_clusters = len(clusters)
    cluster_idx = np.zeros_like(nodes) # map each node to its cluster
    cluster_label = np.zeros(num_clusters, dtype=np.int32) # the class label for each cluster
    for i in range(num_clusters):
        cluster = clusters[i]
        cluster_idx[cluster] = i
        cluster_label[i] = labels[cluster[0]]
    
    # compute an area for each cluster
    cluster_areas = np.zeros(num_clusters)
    np.add.at(cluster_areas, cluster_idx, node_areas)
    
    # determine cluster adjancency
    e0 = cluster_idx[edges[~edge_mask, 0]]
    e1 = cluster_idx[edges[~edge_mask, 1]]
    c_edges = np.stack((e0,  e1), axis=1)
    unq, _ = trimesh.grouping.unique_rows(c_edges)
    c_edges = c_edges[unq]
    
    # for each cluster, compute total area of neighboring clusters by class
    cluster_neighbor_areas = np.zeros((num_clusters, num_classes))
    ind = (c_edges[:,0], cluster_label[c_edges[:,1]])
    np.add.at(cluster_neighbor_areas, ind, cluster_areas[c_edges[:,1]])
    
    # for clusters below threshold, re-assign class to that of max neighboring class area
    small = (cluster_areas < threshold)
    small_labels = np.argmax(cluster_neighbor_areas[small], axis=1)
    cluster_label[small] = small_labels
    
    # map cluster labels back to vertex labels
    for i in range(num_clusters):
        cluster = clusters[i]
        labels[cluster] = cluster_label[i]
    
    return labels
