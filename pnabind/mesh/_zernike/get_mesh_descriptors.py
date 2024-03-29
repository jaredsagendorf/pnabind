import numpy as np

"""
Compute the Zernike moments of a collection of points.


Authors:
    - Arthur Mikhno, 2013, Columbia University (original MATLAB code)
    - Brian Rossa, 2013, Tank Think Labs, LLC (port to Python)
    - Arno Klein, 2013  (arno@mindboggle.info)  http://binarybottle.com

Copyright 2013,  Mindboggle team (http://mindboggle.info), Apache v2.0 License

"""

def meshDescriptors(points, faces, order=10, scale_input=True, center_input=True, zernike_descriptors=True, geometric_moment_invariants=True, parallel=False):
    if parallel:
        from .pipelines import DefaultSerialPipeline as Pipeline
    else:
        from .pipelines import DefaultParallelPipeline as Pipeline
    
    if center_input:
        center = np.mean(points, axis=0)
        points = points - center
    
    if scale_input:
        maxd = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points /= maxd
    
    # compute geometric moments
    pl = Pipeline()
    G = pl.geometric_moments_exact(points, faces, order)
    
    rargs = []
    if zernike_descriptors:
        Z = pl.zernike(G, order)
        descriptors = pl.feature_extraction(Z, order)
        rargs.append(descriptors)
    
    if geometric_moment_invariants:
        gmi = geometricMomentInvariants(G)
        rargs.append(gmi)
    
    return rargs

# def meshZernikeDescriptors(points, faces, order=10, scale_input=True):
    # """
    # Compute the Zernike moments of a surface patch of points and faces.
    
    # Parameters
    # ----------
    # points : list of lists of 3 floats
        # x,y,z coordinates for each vertex
    # faces : list of lists of 3 integers
        # each list contains indices to vertices that form a triangle on a mesh
    # order : integer
        # order of the moments being calculated
    # scale_input : bool
        # translate and scale each object so it is bounded by a unit sphere?
        # (this is the expected input to zernike_moments())
    
    # Returns
    # -------
    # descriptors : list of floats
        # Zernike descriptors

    # Examples
    # --------
    # >>> # Example 1: simple cube (decimation results in a Segmentation Fault):
    # >>> import numpy as np
    # >>> from mindboggle.shapes.zernike.zernike import zernike_moments
    # >>> points = [[0,0,0], [1,0,0], [0,0,1], [0,1,1],
    # ...           [1,0,1], [0,1,0], [1,1,1], [1,1,0]]
    # >>> faces = [[0,2,4], [0,1,4], [2,3,4], [3,4,5], [3,5,6], [0,1,7]]
    # >>> order = 3
    # >>> scale_input = True
    # >>> descriptors = zernike_moments(points, faces, order, scale_input)
    # >>> [np.float("{0:.{1}f}".format(x, 5)) for x in descriptors]
    # [0.09189, 0.09357, 0.04309, 0.06466, 0.0382, 0.04138]
    
    # Example 2: Twins-2-1 left postcentral pial surface -- NO decimation:
               # (zernike_moments took 142 seconds for order = 3 with no decimation)
    
    # >>> from mindboggle.shapes.zernike.zernike import zernike_moments
    # >>> from mindboggle.mio.vtks import read_vtk
    # >>> from mindboggle.guts.mesh import keep_faces
    # >>> from mindboggle.mio.fetch_data import prep_tests
    # >>> urls, fetch_data = prep_tests()
    # >>> label_file = fetch_data(urls['left_freesurfer_labels'], '', '.vtk')
    # >>> points, f1,f2, faces, labels, f3,f4,f5 = read_vtk(label_file)
    # >>> I22 = [i for i,x in enumerate(labels) if x==1022] # postcentral
    # >>> faces = keep_faces(faces, I22)
    # >>> order = 3
    # >>> scale_input = True
    # >>> decimate_fraction = 0
    # >>> decimate_smooth = 0
    # >>> verbose = False
    # >>> descriptors = zernike_moments(points, faces, order, scale_input,
    # ...     decimate_fraction, decimate_smooth, verbose)
    # >>> [np.float("{0:.{1}f}".format(x, 5)) for x in descriptors]
    # [0.00471, 0.0084, 0.00295, 0.00762, 0.0014, 0.00076]

    # Example 3: left postcentral + pars triangularis pial surfaces:

    # >>> from mindboggle.mio.vtks import read_vtk, write_vtk
    # >>> points, f1,f2, faces, labels, f3,f4,f5 = read_vtk(label_file)
    # >>> I20 = [i for i,x in enumerate(labels) if x==1020] # pars triangularis
    # >>> I22 = [i for i,x in enumerate(labels) if x==1022] # postcentral
    # >>> I22.extend(I20)
    # >>> faces = keep_faces(faces, I22)
    # >>> order = 3
    # >>> scale_input = True
    # >>> decimate_fraction = 0
    # >>> decimate_smooth = 0
    # >>> verbose = False
    # >>> descriptors = zernike_moments(points, faces, order, scale_input,
    # ...     decimate_fraction, decimate_smooth, verbose)
    # >>> [np.float("{0:.{1}f}".format(x, 5)) for x in descriptors]
    # [0.00586, 0.00973, 0.00322, 0.00818, 0.0013, 0.00131]

    # View both segments (skip test):

    # >>> from mindboggle.mio.plots import plot_surfaces # doctest: +SKIP
    # >>> from mindboggle.mio.vtks import rewrite_scalars # doctest: +SKIP
    # >>> scalars = -1 * np.ones(np.shape(labels)) # doctest: +SKIP
    # >>> scalars[I22] = 1 # doctest: +SKIP
    # >>> rewrite_scalars(label_file, 'test_two_labels.vtk', scalars,
    # ...                 'two_labels', scalars) # doctest: +SKIP
    # >>> plot_surfaces(vtk_file) # doctest: +SKIP

    # """
    # from .pipelines import DefaultPipeline as Pipeline
    
    # # ------------------------------------------------------------------------
    # # Translate all points so that they are centered at their mean,
    # # and scale them so that they are bounded by a unit sphere:
    # # ------------------------------------------------------------------------
    # if scale_input:
        # center = np.mean(points, axis=0)
        # points = points - center
        # maxd = np.max(np.sqrt(np.sum(points**2, axis=1)))
        # points /= maxd
    
    # # ------------------------------------------------------------------------
    # # Multiprocessor pipeline:
    # # ------------------------------------------------------------------------
    # pl = Pipeline()
    
    # # ------------------------------------------------------------------------
    # # Geometric moments:
    # # ------------------------------------------------------------------------
    # G = pl.geometric_moments_exact(points, faces, order)
    
    # # ------------------------------------------------------------------------
    # # ------------------------------------------------------------------------
    # Z = pl.zernike(G, order)
    
    # # ------------------------------------------------------------------------
    # # Extract Zernike descriptors:
    # # ------------------------------------------------------------------------
    # descriptors = pl.feature_extraction(Z, order).tolist()
    
    # return descriptors

#def meshGeometricMomentInvariants(points, faces, center_input=True):
def geometricMomentInvariants(G):
    
    # from .pipelines import DefaultPipeline as Pipeline
    
    # # ------------------------------------------------------------------------
    # # Translate all points so that they are centered at their mean
    # # ------------------------------------------------------------------------
    # if center_input:
        # center = np.mean(points, axis=0)
        # points = points - center
    
    # # ------------------------------------------------------------------------
    # # Multiprocessor pipeline:
    # # ------------------------------------------------------------------------
    # pl = Pipeline()
    
    # # ------------------------------------------------------------------------
    # # Geometric moments:
    # # ------------------------------------------------------------------------
    # order = 4
    # G = pl.geometric_moments_exact(points, faces, order)
    
    
    # compute invariants
    I1 = (G[4,0,0] + G[0,4,0] + G[0,0,4] + 2*G[2,2,0] + 2*G[2,0,2] + 2*G[0,2,2])/( np.abs(G[0,0,0])**(7/3) + 1e-8 )
    
    I2 = (G[4,0,0]*(G[0,4,0] + G[0,0,4]) + G[0,0,4]*G[0,4,0]
            + 3*(G[2,2,0]**2 + G[2,0,2]**2 + G[0,2,2]**2)
            - 4*(G[1,0,3]*G[3,0,1] +G[1,3,0]*G[3,1,0] + G[0,1,3]*G[0,3,1])
            + 2*(G[0,2,2]*G[2,0,2] + G[0,2,2]*G[2,2,0] + G[2,2,0]*G[2,0,2] + G[0,2,2]*G[4,0,0] + G[0,0,4]*G[2,2,0] + G[0,4,0]*G[2,0,2])
            - 4*(G[1,0,3]*G[1,2,1] + G[1,3,0]*G[1,1,2] + G[0,1,3]*G[2,1,1] + G[1,2,1]*G[3,0,1] + G[1,1,2]*G[3,1,0] + G[2,1,1]*G[0,3,1])
            + 4*(G[2,1,1]**2 + G[1,1,2]**2 + G[1,2,1]**2)
            )/( np.abs(G[0,0,0])**(14/3) + 1e-8 )
    
    I3 = (G[4,0,0]**2 + G[0,4,0]**2 + G[0,0,4]**2 +
            + 4*(G[1,3,0]**2 + G[1,0,3]**2 + G[0,1,3]**2 + G[0,3,1]**2 + G[3,1,0]**2 + G[3,0,1]**2)
            + 6*(G[2,2,0]**2  + G[2,0,2]**2 + G[0,2,2]**2)
            + 12*(G[1,1,2]**2 + G[1,2,1]**2 + G[2,1,1]**2)
            )/( np.abs(G[0,0,0])**(14/3) + 1e-8 )
    
    I4 = (
        G[3,0,0]**2 +
        G[0,3,0]**2 +
        G[0,0,3]**2 +
        3*G[1,2,0]**2 +
        3*G[1,0,2]**2 +
        3*G[0,1,2]**2 +
        3*G[2,1,0]**2 +
        3*G[0,2,1]**2 +
        3*G[2,0,1]**2 +
        6*G[1,1,1]**2
        )/( G[0,0,0]**4 )
    
    I5 = (G[3,0,0]**2  + G[0,3,0]**2 + G[0,0,3]**2
            + G[1,2,0]**2 + G[0,1,2]**2 + G[1,0,2]**2 + G[2,1,0]**2 + G[0,2,1]**2 + G[2,0,1]**2
            + 2*(G[3,0,0]*G[1,2,0] + G[3,0,0]*G[1,0,2] + G[1,2,0]*G[1,0,2]
                + G[0,0,3]*G[2,0,1] + G[0,0,3]*G[0,2,1] + G[0,2,1]*G[2,0,1]
                + G[0,3,0]*G[0,1,2] + G[0,3,0]*G[2,1,0] + G[0,1,2]*G[2,1,0])
            )/( G[0,0,0]**4 +1e-8 )
    
    I6 = (G[2,0,0]*(G[4,0,0] + G[2,2,0] + G[2,0,2])
            + G[0,2,0]*(G[2,2,0] + G[0,4,0] + G[0,2,2])
            + G[0,0,2]*(G[2,0,2] + G[0,2,2] + G[0,0,4])
            + 2*G[1,1,0]*(G[3,1,0] + G[1,3,0] + G[1,1,2])
            + 2*G[1,0,1]*(G[3,0,1] + G[1,2,1] + G[1,0,3])
            + 2*G[0,1,1]*(G[2,1,1] + G[0,3,1] + G[0,1,3])
            )/( G[0,0,0]**4 +1e-8 )
    
    return np.array([I1, I2, I3, I4, I5, I6])
