import numpy as np

def laplacianSmoothing(mesh, X, add_self_loops=True, iterations=1):
    # edges with self-loops
    if add_self_loops:
        Es = np.concatenate([
            mesh.edges,
            np.tile(np.arange(len(X)).reshape(-1,1), (1,2))
        ])
    else:
        Es = mesh.edges
    
    # vertex degrees
    d = np.zeros(len(X))
    np.add.at(d, Es[:, 1], np.ones(len(Es)))
    if X.ndim > 1:
        d = d.reshape(-1, 1)
    
    for i in range(iterations):
        # store smoothed features
        Xs = np.zeros_like(X)
        
        # perform scatter ops
        np.add.at(Xs, Es[:, 1], X[Es[:, 0]])
        
        # divide by degree
        X = Xs / d
    
    return X
