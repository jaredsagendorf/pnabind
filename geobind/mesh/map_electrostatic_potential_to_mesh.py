# third party modules
import numpy as np

# geobind modules
from geobind.utils import generateUniformSpherePoints
from geobind.utils import clipOutliers

def mapElectrostaticPotentialToMesh(mesh, phi, acc, sphere_average=True, efield=False, npts=50, sphere_radius=1.0):
    
    feature_names = []
    features = []
    if(sphere_average):
        # compute point cloud
        kernel = generateUniformSpherePoints(npts, r=sphere_radius) # unit sphere at the origin
        
        # sample over kernel
        V = mesh.vertices
        nV = len(V)
        points = (V[:, np.newaxis] + kernel).reshape(-1, 3) # V*K x 3 array of points
        phi_s = phi(points).reshape(nV, -1) # V x K potential samples
        acc_s = acc(points).reshape(nV, -1) # V x K accessibility samples
        phi_s = phi_s*acc_s # masking inaccessible potential values
        phi_s = phi_s.sum(axis=1)/acc_s.sum(axis=1) # V array of averaged potential
        
        feature_names.append('averaged_potential')
        features.append(clipOutliers(phi_s))
        
    
    return np.array(features).T, feature_names
