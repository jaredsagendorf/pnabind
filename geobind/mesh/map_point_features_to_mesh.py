# third party modules
import numpy as np

def wfn(dist, cutoff, offset=0, weight_method='inverse_distance', minw=0.5, maxw=1.0, ):
    if(minw >= maxw):
        raise ValueError("minw must be < maxw!")
    
    # decide how we weight by distance
    if(weight_method == 'inverse_distance'):
        b = minw/(maxw - minw)
        a = maxw*b
        u = (dist-offset)/(cutoff-offset)
        return np.clip(a/(u + b), minw, maxw)
    elif(weight_method == 'linear'):
        b = maxw
        a = -(maxw - minw)
        u = (dist-offset)/(cutoff-offset)
        return np.clip(a*u + b, minw, maxw)
    elif(weight_method == 'binary'):
        return np.ones(d.size)
    else:
        raise ValueError("Unknown value of argument `weight_method`: {}".format(weight_method))
    

def mapPointFeaturesToMesh(mesh, points, features, distance_cutoff=3.0, offset=None, map_to='neighborhood', weight_method='inverse_distance', **kwargs):
    
    X = np.zeros((mesh.num_vertices, features.shape[1])) # store the mapped features
    W = np.zeros(mesh.num_vertices) # weights determined by distance from points to vertices
    if(offset == None):
        offset = np.zeros(len(points))
    assert len(points) == len(features) and len(points) == len(offset)
    
    for i in range(len(points)):
        p = points[i]
        f = features[i]
        o = offset[i]
        
        # decide how to map point features to vertices
        if(map_to == 'neighborhood'):
            # map features to all vertices within a neighborhood, weighted by distance
            v, d = mesh.verticesInBall(p, distance_cutoff)
        elif(map_to == 'nearest'):
            # get the neartest vertex
            v, d = mesh.nearestVertex(p)
        else:
            raise ValueError("Unknown value of argument `map_to`: {}".format(map_to))
    
        if(len(v) > 0):
            w = wfn(d, distance_cutoff, o, weight_method, **kwargs)
            X[v] += np.outer(w, f)
            W[v] += w
    
    # set zero weights to 1
    wi = (W == 0)
    W[wi] = 1.0
    
    # scale by weights
    X /= W.reshape(-1, 1)
    
    return X
