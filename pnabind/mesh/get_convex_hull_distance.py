# third party modules
import numpy as np

def getConvexHullDistance(mesh):
    try:
        import igl
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The dependency 'igl' is required for this functionality!")
    
    hull = mesh.convex_hull
    distances, indices, closest_pts = igl.point_mesh_squared_distance(mesh.vertices, hull.vertices, hull.faces)
    
    mesh.vertex_attributes['convex_hull_distance'] = np.sqrt(distances)
    
    return ['convex_hull_distance']
