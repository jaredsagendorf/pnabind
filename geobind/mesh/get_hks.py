# third party modules
from scipy.sparse.linalg import eigsh as sp_eigs
from scipy.sparse import diags
import numpy as np

# geobind modules
from geobind.utils import clipOutliers

def getHKS(mesh, num_samples=3, num_components=50, feature_name='hks', tau=1, eps=1e-5, normalize=True, **kwargs):
    
    # compute eigenvalues and eigenvectors of Laplace-Beltrami operator
    L = -mesh.cot_matrix
    M = mesh.mass_matrix
    
    try:
        evals, evecs = sp_eigs(L, k=num_components, M=M,  which='LM', sigma=0, **kwargs)
    except RuntimeError:
        # add small value to cot matrix to try and make it invertable
        D = diags(eps*np.ones(L.shape[0]))
        L = L + D
        evals, evecs = sp_eigs(L, k=num_components, M=M,  which='LM', sigma=0, **kwargs)
    
    if normalize:
        evals = evals/evals.sum()
        scale = mesh.area
    else:
        scale = 1
    
    # determine time samples
    tmin = tau/min(evals.max(), 1e+1)
    tmax = tau/max(evals.min(), 1e-3)
    tsamps = np.exp(np.linspace(np.log(tmin), np.log(tmax), num_samples))
    
    # compute heat kernel signatures
    evecs = evecs**2
    feature_names = []
    for i, t in enumerate(tsamps):
        fn = "{}{}".format(feature_name, i+1)
        HKS = scale*np.sum(np.exp(-t*evals)*evecs, axis=1)
        mesh.vertex_attributes[fn] = clipOutliers(HKS)
        feature_names.append(fn)
    
    return feature_names 
