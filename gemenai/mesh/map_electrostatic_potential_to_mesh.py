# third party modules
import numpy as np

# gemenai modules
from gemenai.utils import generateUniformSpherePoints
from gemenai.utils import clipOutliers
from .laplacian_smoothing import laplacianSmoothing

PHI_COEF = [0.1421438807766923, 1.6245703034444483] # linear coefficients to scale phi
DPH_COEF = [1.4502167585912877, -5.405648765703955] # linear coefficients to scale dphi

def mapElectrostaticPotentialToMesh(mesh, phi, acc, 
        sphere_average=True,
        npts=50,
        sphere_radius=1.0,
        efield=False,
        diff_method='symmetric_difference',
        h=None,
        laplace_smooth=False,
        scale_to_tabi=False
    ):
    
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
        
        phi_array = clipOutliers(phi_s)
    else:
        phi_array = clipOutliers(phi(V))
    
    if scale_to_tabi:
        phi_array = PHI_COEF[1]*phi_array + PHI_COEF[0]
    
    features.append(phi_array)
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
            Ex = (phi(V+dx) - phi(V-dx))/(2*h[0])
            Ey = (phi(V+dy) - phi(V-dy))/(2*h[1])
            Ez = (phi(V+dz) - phi(V-dz))/(2*h[2])
        elif diff_method == 'five_point_stencil':
            Ex = (-phi(V+2*dx) + 8*phi(V+dx) - 8*phi(V-dx) + phi(V-2*dx))/(12*h[0])
            Ey = (-phi(V+2*dy) + 8*phi(V+dy) - 8*phi(V-dy) + phi(V-2*dy))/(12*h[1])
            Ez = (-phi(V+2*dz) + 8*phi(V+dz) - 8*phi(V-dz) + phi(V-2*dz))/(12*h[2])
        else:
            raise ValueError("Unknown value of parameter `diff_method`: '{}'".format(diff_method))
        
        #if sphere_average:
        #    Ex = (Ex.reshape(nV, -1)*pts_mask).sum(axis=1)/pts_msum
        #    Ey = (Ey.reshape(nV, -1)*pts_mask).sum(axis=1)/pts_msum
        #    Ez = (Ez.reshape(nV, -1)*pts_mask).sum(axis=1)/pts_msum
        
        sig = -N[:,0]*Ex - N[:,1]*Ey - N[:,2]*Ez
        sig = clipOutliers(sig)
        if laplace_smooth:
            sig = laplacianSmoothing(mesh, sig, iterations=2)
        
        if scale_to_tabi:
            sig = DPH_COEF[1]*sig + DPH_COEF[0]
        
        features.append(sig)
        feature_names.append('efield_projection')
    
    return np.array(features).T, feature_names
