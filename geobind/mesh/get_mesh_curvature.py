# third party modules
import numpy as np

# geobind modules
from geobind.utils import clipOutliers

def getMeshCurvature(mesh, gaussian_curvature=True, mean_curvature=True, shape_index=True, remove_outliers=True):
    try:
        import igl
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The dependency 'igl' is required for this functionality!")
    
    v1, v2, k1, k2 = igl.principal_curvature(mesh.vertices, mesh.faces)
    
    k1 = clipOutliers(k1)
    k2 = clipOutliers(k2)
    
    feature_names = []
    features = []
    if gaussian_curvature:
        feature_names.append('gaussian_curvature')
        mesh.vertex_attributes['gaussian_curvature'] = k1*k2
    
    if mean_curvature:
        feature_names.append('mean_curvature')
        mesh.vertex_attributes['mean_curvature'] = (k1 + k2)/2
    
    if shape_index:
        shape_index = -2*np.arctan( (k1 + k2)/(k1 - k2) )/np.pi
        feature_names.append('shape_index')
        mesh.vertex_attributes['shape_index'] = shape_index
    
    return feature_names
