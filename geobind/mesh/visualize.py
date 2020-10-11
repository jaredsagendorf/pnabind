# third party modules
import trimesh
import numpy

def visualizeMesh(mesh, data=None, color_map='seismic', shift_axis='x', **kwargs):
    if(data is None):
        return mesh.show(**kwargs)
    
    scene = [] # hold a list of meshes that determine the scene
    si = ['x', 'y', 'z'].index(shift_axis)
    offset = 0.0 # cumulative distance we translate new meshes
    shift = np.array([0.0, 0.0, 0.0]) # direction to translate meshes
    shift[si] = 1.0
    for i in range(data.ndim):
        # add meshes to scene list
        m = mesh.copy()
        m.apply_translation(offset*shift)
        m.visual.vertex_colors = trimesh.visual.interpolate(data[:,i], color_map=color_map)
        offset += m.bounding_box.extents[si]
        meshes.append(m)
    return trimesh.Scene(scene).show(**kwargs)
