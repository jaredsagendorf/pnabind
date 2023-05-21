# third party modules
import numpy as np

# gemenai modules
from gemenai.utils import segmentsIntersectTriangles

def getClassSegmentations(edges, labels, 
            faces=None,
            area_faces=None,
            merge_nearby_clusters=False,
            vertices=None,
            merge_distance=10.0,
            no_merge=[],
            check_mesh_intersection=True,
            return_vertex_areas=False,
            return_boundary_vertices=False
    ):
    """This function identifies clusters of connected vertices which share a common class label and
    returns them as a list of vertex indices as well as arrays containing the area of each cluster, 
    label of each clusters, and map from vertex indicies to cluster index"""
    
    try:
        import trimesh
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The dependency 'Trimesh' is required for this functionality!")
    try:
        from scipy.spatial.distance import cdist
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The dependency 'SciPy' is required for this functionality!")
    
    nodes = np.arange(len(labels))
    
    # compute a weight for each node
    if faces is not None and area_faces is not None:
        # use face areas to assign node weights
        vertex_areas = np.zeros_like(nodes, dtype=np.float32)
        np.add.at(vertex_areas, faces[:, 0], area_faces/3)
        np.add.at(vertex_areas, faces[:, 1], area_faces/3)
        np.add.at(vertex_areas, faces[:, 2], area_faces/3)
    else:
        # equal weighting for every node
        vertex_areas = np.ones_like(nodes, dtype=np.float32)
    
    # determine connected components based on class and adjacency and assign each cluster to a 
    # class label
    edge_mask = (labels[edges[:,0]] == labels[edges[:,1]]) # edges where both nodes agree on class 
    
    # clusters of connected vertices that have same class. Returns a list of arrays that contain indices into labels
    clusters = trimesh.graph.connected_components(edges[edge_mask], nodes=nodes) 
    
    # possibly merge clusters
    if merge_nearby_clusters:
        assert vertices is not None
        assert faces is not None
        
        # collect clusters of the same class
        cluster_sets = {}
        for cluster in clusters:
            cls = int(labels[cluster[0]]) # class of this cluster
            if cls not in cluster_sets:
                cluster_sets[cls] = []
            
            # get cluster center
            #cm = vertices[cluster].mean(axis=0)
            #cm_dist = np.linalg.norm(vertices[cluster] - cm, axis=1)
            #ci = cluster[np.argmin(cm_dist)]
            
            # add cluster to class set
            #cluster_sets[cls].append((cluster, vertices[ci]))
            cluster_sets[cls].append(cluster)
        
        # for each class, create a graph where we add an edge if two clusters
        # are within a threshold distance that respects line of sight
        new_clusters = []
        tri = (vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]])
        for cls in cluster_sets:
            cluster_set = cluster_sets[cls]
            N = len(cluster_set)
            cedges = []
            
            # check pair-wise distances
            if cls not in no_merge:
                for i in range(N):
                    #cluster_i, center_i = cluster_set[i]
                    for j in range(i+1, N):
                        
                        # get minimum distance between cluster i and j
                        D = cdist(vertices[cluster_set[i]], vertices[cluster_set[j]])
                        imin, jmin = np.unravel_index(D.argmin(), D.shape)
                        min_dist = D[imin, jmin]
                        
                        if min_dist <= merge_distance:
                            vi = cluster_set[i][imin] # vertex index in cluster i
                            vj = cluster_set[j][jmin] # vertex index in cluster j
                            
                            if check_mesh_intersection:
                                seg = (vertices[vi].reshape(1, -1), vertices[vj].reshape(1, -1)) # line segement joining vertex vi and vj
                                counts = segmentsIntersectTriangles(seg, tri) # check if this segment intersects the mesh
                                intersects = (counts > 0)
                                if len(intersects) == 1:
                                    cedges.append([i, j]) # no intersection, join these clusters
                            else:
                                cedges.append([i, j])
            
            # get connected components
            cedges = np.array(cedges)
            cnodes = np.arange(N)
            connected_clusters = trimesh.graph.connected_components(cedges, nodes=cnodes)
            
            # merge clusters
            for cc in connected_clusters:
                new_clusters.append(np.concatenate([cluster_set[i] for i in cc], axis=-1))
        
        clusters = new_clusters
    
    num_clusters = len(clusters)
    cluster_idx = np.zeros_like(nodes) # map each node to its cluster
    cluster_labels = np.zeros(num_clusters, dtype=np.int32) # the class label for each cluster
    for i in range(num_clusters):
        cluster = clusters[i]
        cluster_idx[cluster] = i
        cluster_labels[i] = labels[cluster[0]] # use label of first vertex in cluster
    
    # compute an area for each cluster based on node weights
    cluster_areas = np.zeros(num_clusters)
    np.add.at(cluster_areas, cluster_idx, vertex_areas)
    
    args = [clusters, cluster_idx, cluster_areas, cluster_labels]
    if return_vertex_areas:
        args.append(vertex_areas)
    
    if return_boundary_vertices:
        boundary_vertices = set(edges[~edge_mask].flatten())
        cluster_boundaries = []
        for c in clusters:
            c_vertices = set(c)
            boundary = boundary_vertices.intersection(c_vertices)
            cluster_boundaries.append(np.array(list(boundary), dtype=np.int32))
        args.append(cluster_boundaries)
    
    return tuple(args)
