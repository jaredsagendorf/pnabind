# third party modules
import numpy as np

# geobind modules
from geobind.utils import generateUniformSpherePoints
from geobind.utils import clipOutliers

def mapElectrostaticPotentialToMesh(mesh, phi, acc, sphere_average=True, npts=50, sphere_radius=1.0, efield=False, diff_method='symmetric_difference', h=None):
    
    feature_names = []
    features = []
    V = mesh.vertices
    N = mesh.vertex_normals
    nV = len(V)
    
    # Determine what points to sample
    if sphere_average:
        # compute point cloud
        kernel = generateUniformSpherePoints(npts, r=sphere_radius) # unit sphere at the origin
        
        # sample over kernel
        points = (V[:, np.newaxis] + kernel).reshape(-1, 3) # V*K x 3 array of points
        
        # accessibility mask
        pts_mask = acc(points).reshape(nV, -1) # V x K accessibility samples
        pts_msum = pts_mask.sum(axis=1) # V array of summed mask
    else:
        points = V
    
    # Map electrostatic potential
    if sphere_average:
        phi_s = phi(points).reshape(nV, -1) # V x K potential samples
        phi_s = phi_s*pts_mask # masking inaccessible potential values
        phi_s = phi_s.sum(axis=1)/pts_msum # V array of averaged potential
        
        features.append(clipOutliers(phi_s))
    else:
        features.append(clipOutliers(phi(V)))
    feature_names.append('averaged_potential')
    
    if efield:
        # Map electric field to vertex normals
        if h is None:
            h = phi.grid.delta / 5
        elif isinstance(h, float):
            h = np.array([h, h, h])
        dx = h[0]*np.array([1, 0, 0])
        dy = h[1]*np.array([0, 1, 0])
        dz = h[2]*np.array([0, 0, 1])
        
        if diff_method == 'symmetric_difference':
            Ex = (phi(points+dx) - phi(points-dx))/(2*h[0])
            Ey = (phi(points+dy) - phi(points-dy))/(2*h[1])
            Ez = (phi(points+dz) - phi(points-dz))/(2*h[2])
        elif diff_method == 'five_point_stencil':
            Ex = (-phi(points+2*dx) + 8*phi(points+dx) - 8*phi(points-dx) + phi(points-2*dx))/(12*h[0])
            Ey = (-phi(points+2*dy) + 8*phi(points+dy) - 8*phi(points-dy) + phi(points-2*dy))/(12*h[1])
            Ez = (-phi(points+2*dz) + 8*phi(points+dz) - 8*phi(points-dz) + phi(points-2*dz))/(12*h[2])
        else:
            raise ValueError("Unknown value of parameter `diff_method`: '{}'".format(diff_method))
        
        if sphere_average:
            Ex = (Ex.reshape(nV, -1)*pts_mask).sum(axis=1)/pts_msum
            Ey = (Ey.reshape(nV, -1)*pts_mask).sum(axis=1)/pts_msum
            Ez = (Ez.reshape(nV, -1)*pts_mask).sum(axis=1)/pts_msum
        
        sig = -N[:,0]*Ex - N[:,1]*Ey - N[:,2]*Ez
        features.append(clipOutliers(sig))
        feature_names.append('efield_projection')
    
    return np.array(features).T, feature_names
