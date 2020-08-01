# third party modules
import numpy as np
import igl

def getConvexHullDistance(mesh):
    hull = mesh.convex_hull
    distances, indices, closest_pts = igl.point_mesh_squared_distance(mesh.vertices, hull.vertices, hull.faces)
    
    mesh.vertex_attributes['convex_hull_distance'] = np.sqrt(distances)
    
    return ['convex_hull_distance']
