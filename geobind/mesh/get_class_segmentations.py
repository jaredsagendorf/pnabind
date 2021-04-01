import numpy as np
import trimesh
from scipy.spatial.distance import cdist

def getClassSegmentations(edges, labels, faces=None, area_faces=None, merge_nearby_clusters=False, vertices=None, merge_distance=100.0, no_merge=[]):
    nodes = np.arange(len(labels))
    
    # compute an area for each node
    if faces is not None and area_faces is not None:
        # use face areas to assign node weights
        node_areas = np.zeros_like(nodes, dtype=np.float32)
        np.add.at(node_areas, faces[:, 0], area_faces/3)
        np.add.at(node_areas, faces[:, 1], area_faces/3)
        np.add.at(node_areas, faces[:, 2], area_faces/3)
    else:
        # equal weighting for every node
        node_areas = np.ones_like(nodes, dtype=np.float32)
    
    # determine connected components based on class and adjacency and assign each cluster to a 
    # class label
    edge_mask = (labels[edges[:,0]] == labels[edges[:,1]]) # edges where both nodes agree on class 
    clusters = trimesh.graph.connected_components(edges[edge_mask], nodes=nodes) # clusters of connected vertices that have same class
    if merge_nearby_clusters:
        assert vertices is not None
        
        # collect clusters of the same class
        cluster_sets = {}
        for cluster in clusters:
            cl = int(labels[cluster[0]]) # class of this cluster
            if cl not in cluster_sets:
                cluster_sets[cl] = []
            cluster_sets[cl].append(cluster)
        
        # for each class, create a graph where we add an edge if two clusters
        # are within a threshold distance
        new_clusters = []
        for cl in cluster_sets:
            N = len(cluster_sets[cl])
            cedges = []
            
            # check distances
            if cl not in no_merge:
                for i in range(N):
                    for j in range(i+1, N):
                        mdist = cdist(vertices[cluster_sets[cl][i]], vertices[cluster_sets[cl][j]]).min()
                        if mdist <= merge_distance:
                            cedges.append([i, j])
            
            # get connected components
            cedges = np.array(cedges)
            cnodes = np.arange(N)
            connected_clusters = trimesh.graph.connected_components(cedges, nodes=cnodes)
            
            # merge clusters
            for cc in connected_clusters:
                new_clusters.append(np.concatenate([cluster_sets[cl][i] for i in cc], axis=-1))
        
        clusters = new_clusters
    
    num_clusters = len(clusters)
    cluster_idx = np.zeros_like(nodes) # map each node to its cluster
    cluster_labels = np.zeros(num_clusters, dtype=np.int32) # the class label for each cluster
    for i in range(num_clusters):
        cluster = clusters[i]
        cluster_idx[cluster] = i
        cluster_labels[i] = labels[cluster[0]] # use label of first vertex in cluster
    
    # compute an area for each cluster
    cluster_areas = np.zeros(num_clusters)
    np.add.at(cluster_areas, cluster_idx, node_areas)
    
    return clusters, cluster_idx, cluster_areas, cluster_labels
