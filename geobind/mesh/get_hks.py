# third party modules
from scipy.sparse.linalg import eigsh as sp_eigs
import numpy as np

# geobind modules
from geobind.utils import clipOutliers

def getHKS(mesh, num_samples=3, num_components=50, feature_name='hks', tau=1, **kwargs):
    
    # compute eigenvalues and eigenvectors of Laplace-Beltrami operator
    L = -mesh.cot_matrix
    M = mesh.mass_matrix
    
    evals, evecs = sp_eigs(L, k=num_components, M=M,  which='LM', sigma=0, **kwargs)
    
    # determine time samples
    tmin = tau/min(evals.max(), 1e+1)
    tmax = tau/max(evals.min(), 1e-3)
    tsamps = np.exp(np.linspace(np.log(tmin), np.log(tmax), num_samples))
    
    # compute heat kernel signatures
    evecs = evecs**2
    feature_names = []
    for i, t in enumerate(tsamps):
        fn = "{}{}".format(feature_name, i+1)
        HKS = np.sum(np.exp(-t*evals)*evecs, axis=1)
        mesh.vertex_attributes[fn] = clipOutliers(HKS)
        feature_names.append(fn)
    
    return feature_names 
