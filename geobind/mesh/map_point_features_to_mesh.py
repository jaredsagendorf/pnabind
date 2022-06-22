# third party modules
import numpy as np

# geobind modules
from geobind.utils import clipOutliers
from .laplacian_smoothing import laplacianSmoothing

def wfn(dist, cutoff, offset=0, weight_method='inverse_distance', minw=0.5, maxw=1.0):
    if minw >= maxw:
        raise ValueError("minw must be < maxw!")
    
    # decide how we weight by distance
    if weight_method == 'inverse_distance':
        b = maxw/(minw - maxw)
        a = minw*b
        u = (cutoff - dist + offset)/cutoff
        return np.clip(a/(u + b), minw, maxw)
    elif weight_method == 'linear':
        b = minw
        a = (maxw - minw)
        u = (cutoff - dist + offset)/cutoff
        return np.clip(a*u + b, minw, maxw)
    elif weight_method == 'binary':
        return np.ones(dist.size)
    else:
        raise ValueError("Unknown value of argument `weight_method`: {}".format(weight_method))

def mapPointFeaturesToMesh(mesh, points, features, distance_cutoff=3.0, offset=None, map_to='neighborhood', weight_method='inverse_distance', laplace_smooth=False, clip_values=False, iterations=1, **kwargs):
    
    if clip_values:
        features = clipOutliers(features, axis=0)
    
    # decide how to map point features to vertices
    if map_to == 'neighborhood':
        X = np.zeros((mesh.num_vertices, features.shape[1])) # store the mapped features
        W = np.zeros(mesh.num_vertices) # weights determined by distance from points to vertices
        if offset == None:
            offset = np.zeros(len(points))
        assert len(points) == len(features) and len(points) == len(offset)
        
        for i in range(len(points)):
            p = points[i]
            f = features[i]
            o = offset[i]
            
            # map features to all vertices within a neighborhood, weighted by distance
            t = distance_cutoff + o
            v, d = mesh.verticesInBall(p, t)
            
            if len(v) > 0:
                w = wfn(d, distance_cutoff, o, weight_method, **kwargs)
                X[v] += np.outer(w, f)
                W[v] += w
    
        # set zero weights to 1
        wi = (W == 0)
        W[wi] = 1.0
        
        # scale by weights
        X /= W.reshape(-1, 1)
    elif map_to == 'nearest':
        # get the neartest point to each vertex
        assert len(points) == len(features)
        
        try:
            from scipy.spatial import cKDTree
        except ModuleNotFoundError:
            raise ModuleNotFoundError("The dependency 'SciPy' is required for this functionality!")
        kdt = cKDTree(points)
        
        d, ind = kdt.query(mesh.vertices)
        
        X = features[ind]
    else:
        raise ValueError("Unknown value of argument `map_to`: {}".format(map_to))
    
    if laplace_smooth:
        X = laplacianSmoothing(mesh, X, iterations=iterations)
    
    return X
