import numpy as np
import trimesh

from .get_class_segmentations import getClassSegmentations

def smoothMeshLabels(edges, labels, num_classes, 
            threshold=16.0,
            faces=None,
            area_faces=None,
            ignore_class=None,
            **kwargs
    ):
    # get class segmentations
    clusters, cluster_idx, cluster_areas, cluster_labels = getClassSegmentations(edges, labels, faces=faces, area_faces=area_faces, **kwargs)
    num_clusters = len(clusters)
    edge_mask = (labels[edges[:,0]] == labels[edges[:,1]])
    
    # determine cluster adjancency
    e0 = cluster_idx[edges[~edge_mask, 0]]
    e1 = cluster_idx[edges[~edge_mask, 1]]
    c_edges = np.stack((e0,  e1), axis=1)
    unq, _ = trimesh.grouping.unique_rows(c_edges)
    c_edges = c_edges[unq]
    
    # for each cluster, compute total area of neighboring clusters by class
    cluster_neighbor_areas = np.zeros((num_clusters, num_classes))
    ind = (c_edges[:,0], cluster_labels[c_edges[:,1]])
    np.add.at(cluster_neighbor_areas, ind, cluster_areas[c_edges[:,1]])
    
    # for clusters below threshold, re-assign class to that of max neighboring class area
    small = (cluster_areas < threshold)
    if ignore_class is not None:
        # skip classes we do not want to change
        skip = np.zeros_like(cluster_areas, dtype=bool)
        for c in ignore_class:
            skip += (cluster_labels == c)
        small = small*(~skip)
    small_labels = np.argmax(cluster_neighbor_areas[small], axis=1)
    cluster_labels[small] = small_labels
    
    # map cluster labels back to vertex labels
    for i in range(num_clusters):
        cluster = clusters[i]
        labels[cluster] = cluster_labels[i]
    
    return labels
