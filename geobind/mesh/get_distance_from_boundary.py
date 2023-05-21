import numpy as np
from gemenai.mesh import getClassSegmentations
from gemenai.mesh import getFaceArea

def getDistanceFromBoundary(V, F, Y, E=None, area_faces=None, skip_classes=[0]):
    import gdist
    
    # compute class segmentations
    if E is None:
        E = F[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)) # edge indices
    if area_faces is None:
        area_faces = getFaceArea(V, F)
    clusters, cluster_idx, cluster_areas, cluster_labels, boundaries = getClassSegmentations(E, Y, faces=F, area_faces=area_faces, return_boundary_vertices=True)
    
    # compute distance of every vertex in cluster to the boundary
    mask = np.zeros_like(Y, dtype=bool)
    D = np.zeros_like(Y, dtype=np.float32)
    for i in range(len(clusters)):
        if cluster_labels[i] in skip_classes:
            continue
        d = gdist.compute_gdist(V, F.astype(np.int32), boundaries[i].astype(np.int32), clusters[i].astype(np.int32))
        D[clusters[i]] = d
        mask[clusters[i]] = True
    
    return D, mask
