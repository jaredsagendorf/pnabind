# standard packages
import glob
import os

# third party packages
import numpy as np

# gemenai packages
from .run_nanoshaper import runNanoShaper
from .mesh import Mesh
from .map_point_features_to_mesh import mapPointFeaturesToMesh

def getPockets(atoms, mesh, nn_cutoff=1.5, radius_big=3.0, clean=True, feature_name='pocket', formatstr="{}_{}", **kwargs):
    # run NanoShaper
    runNanoShaper(atoms, "pockets", ".", pockets_only=True, kwargs=dict(radius_big=radius_big))
    
    # gather info in mesh
    points = []
    features = []
    pockets = glob.glob("cav_tri*.off")
    if(len(pockets) > 0):
        for p in pockets:
            pocket = Mesh(p, process=False, remove_disconnected_components=False)
            points.append(pocket.vertices)
            features.append(
                np.tile([
                    pocket.volume,
                    pocket.area,
                    pocket.volume/pocket.area,
                    pocket.aspect_ratio],
                    (pocket.num_vertices, 1)
            ))
        points = np.concatenate(points)
        features = np.concatenate(features)
    
    if(clean):
        # remove all the stuff NanoShaper generated
        files = ["numcav.txt", "cavities.txt", "cavitiesSize.txt", "cavAtomsSerials.txt" "triangulatedSurf.off"]
        files += glob.glob("all_cav*.txt")
        files += glob.glob("cav*.txt")
        files += glob.glob("cav_tri*.off")
        for f in files:
            if(os.path.exists(f)):
                os.remove(f)
    
    feature_names = [
        formatstr.format(feature_name, "volume"),
        formatstr.format(feature_name, "area"),
        formatstr.format(feature_name, "volume/area"),
        formatstr.format(feature_name, "aspect_ratio")
    ]
    if len(points) > 0:
        return mapPointFeaturesToMesh(mesh, points, features, distance_cutoff=nn_cutoff, **kwargs), feature_names
    else:
        return np.zeros((mesh.num_vertices, len(feature_names))), feature_names
