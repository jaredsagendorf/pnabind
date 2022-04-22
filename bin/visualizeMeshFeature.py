#!/usr/bin/env python

# command line args
import argparse
PARSER = argparse.ArgumentParser()
PARSER.add_argument("data_file")
PARSER.add_argument("extras_file", nargs='?')
PARSER.add_argument("--color_map", dest='color_map', default='seismic',
        help="Name of a matplotlib color map.")
PARSER.add_argument("--multiclass_labels", action='store_true',
        help="Use multiclass coloring")
PARSER.add_argument("--smooth", action='store_true',
        help="Turn on smooth shading")
PARSER.add_argument("--point_cloud", action='store_true',
        help="Render Point Clouds")
PARSER.add_argument("--save_scene", action='store_true', default=False,
        help="Save each rendered scene to file as .ply")
PARSER.add_argument("--vmin", type=float, default=None,
        help="Min value for color scale.")
PARSER.add_argument("--vmax", type=float, default=None,
        help="Max value for color scale.")
PARSER.add_argument("--ptp_threshold", type=float, default=None,
        help="Minimum variation for applying color map.")
ARGS = PARSER.parse_args()

# builtin modules
import os
import re

# third party modules
import numpy as np
import trimesh
from matplotlib.pyplot import get_cmap

# binary label colors
binary_colors = np.array([
    [1.00, 1.00, 1.00], # -1: masked
    [0.55, 0.70, 0.40], #  0: non-binding site
    [1.00, 0.50, 0.00], #  1: binding site
    [1.00, 0.00, 0.00], #  2: false positives
    [0.73, 0.33, 0.83], #  3: false negatives
])

# binary label colors
multiclass_colors = np.array([
    [1.00, 1.00, 1.00], # -1: masked
    [0.55, 0.70, 0.40], #  0: non-binding site
    [0.87, 0.63, 0.87], #  1: pink
    [0.00, 1.00, 0.94], #  2: cyan
    [1.00, 0.50, 0.00], #  3: orange
    [1.00, 1.00, 0.00], #  4: yellow
    [0.73, 0.33, 0.83], #  5: purple
    [51/255, 51/255, 1.0], # 6: blue
    [1.00, 0.00, 0.00] # 7: red (false prediction)
])

def visualizeMesh(mesh, data=None, colors=None, color_map='seismic', int_color_map=None, save=True, max_width=4, shift_axis='x', vmin=None, vmax=None, ptp_threshold=0, scene_name="scene", **kwargs):
    # figure out where to get color information
    if data is None and colors is None:
        # just visualize the mesh geometry
        return mesh.show(**kwargs)
    elif colors is None:
        # compute colors from the data array
        vertex_colors = []
        for i in range(len(data)):
            if data[i].dtype == np.int64 or data[i].dtype == bool:
                vertex_colors.append(trimesh.visual.to_rgba(int_color_map[data[i]+1]))
            else:
                if np.ptp(data[i]) <= ptp_threshold:
                    vertex_colors.append(trimesh.visual.to_rgba([0.8, 0.8, 0.8]))
                else:
                    cmap = get_cmap(color_map)
                    vmin = vmin if (vmin is not None) else data[i].min()
                    vmax = vmax if (vmax is not None) else data[i].max()
                    
                    # scale values to 0.0 - 1.0 and get colors
                    colors = cmap(
                        np.clip((data[i] - vmin)/(vmax - vmin), 0.0, 1.0)
                    )
                    rgba = trimesh.visual.to_rgba(colors, dtype=np.uint8)
                    
                    #vertex_colors.append(trimesh.visual.interpolate(data[i], color_map=color_map))
                    vertex_colors.append(rgba)
    else:
        # use the given colors
        if isinstance(colors, list):
            vertex_colors = colors
        else:
            vertex_colors = [colors]
    scene = [] # hold a list of meshes that determine the scene
    
    # determine how to arrange multiple copies of the mesh
    si = ['x', 'y', 'z'].index(shift_axis)
    offset = 0.0 # cumulative distance we translate new meshes
    shift = np.array([0.0, 0.0, 0.0]) # direction to translate meshes
    shift[si] = 1.0
    
    # loop over each mesh copy
    for i in range(len(vertex_colors)):
        # add meshes to scene list
        m = mesh.copy()
        m.apply_translation(offset*shift)
        m.visual.vertex_colors = vertex_colors[i] 
        offset += 1.1*m.bounding_box.extents[si]
        scene.append(m)
    
    if save and len(scene) == 1:
        scene[0].export("{}.ply".format(scene_name), encoding="ascii", vertex_normal=True)
    
    return trimesh.Scene(scene).show(**kwargs)

def getDataArrays(INP):
    arrays = []
    for inp in INP:
        if inp in data:
            arrays.append(data[inp])
        else:
            m = re.match("([A-Za-z]+)(\d*)", inp)
            if m:
                i = int(m.group(2))
                key = m.group(1)
                if key in data:
                    arrays.append(data[key][:,i])
            else:
                for key in data:
                    dtype = data[key].dtype
                    if not any([dtype == np.int64, dtype == np.bool, dtype == np.float32, dtype == np.float64]):
                        continue
                    if re.search(inp, key):
                        arrays.append(data[key])
                        break
    return arrays

def getDataFields(data, key_name=None):
    feature_list = ""
    for key in data:
        if key == "V" or key == "F" or key == "N":
            continue
        dtype = data[key].dtype
        if any([dtype == np.int64, dtype == bool, dtype == np.float32, dtype == np.float64]):
            if data[key].ndim > 1:
                if key_name and key in key_name:
                    for i in range(data[key].shape[1]):
                        feature_list += "{}{:<2d}: {}\n".format(key, i, key_name[key][i])
                else:
                    for i in range(data[key].shape[1]):
                        feature_list += "{}{:<2d}: data field\n".format(key, i)
            else:
                feature_list += "{}: data field\n".format(key)
    
    return feature_list

# Read in data files
data = np.load(ARGS.data_file, allow_pickle=True)
if ARGS.extras_file:
    extras = np.load(ARGS.extras_file, allow_pickle=True)

if ARGS.point_cloud:
    mesh = trimesh.PointCloud(vertices=data['V'], process=False)
else:
    mesh = trimesh.Trimesh(vertices=data['V'], faces=data['F'], process=False)

# list of features in data file
feature_list = ""
#if "feature_names" in data:
    #for i, feature in enumerate(data['feature_names']):
        #feature_list += "X{:<2d}: {}\n".format(i, feature)
if "feature_names" in data:
    feature_list += getDataFields(data, key_name={'X': data['feature_names']})
else:
    feature_list += getDataFields(data)

if ARGS.extras_file:
    feature_list += getDataFields(extras)
feauture_list = feature_list.strip()

# feature array
if("X" in data):
    size = data["X"].shape[1]
else:
    size = -1

# input string
input_string = """
Enter one of the following:
    l to list features
    m to visualize only the mesh
    c to compare two sets of labels
    q to exit
    t a threshold to visualize predictions with: Y = (P1 >= t)
    a list of data fields, separated by space (e.g. "X1 X2 Y")
:""".strip()

### TODO: add selection like "Y X0 X1 YPR"
# main loop
while True:
    inp = input(input_string).strip()
    
    # check what was typed
    INP = inp.strip()
    if(INP == 'q'):
        print("Goodbye.")
        break
    elif(INP == 'l'):
        print(feature_list)
    elif(inp[0] == 't'):
        t = float(inp[1:].strip())
        arrays = getDataArrays(['P1'])
        Y = (arrays[0] >= t).astype(int)
        visualizeMesh(mesh, [Y], color_map=ARGS.color_map, smooth=ARGS.smooth, int_color_map=binary_colors, save=ARGS.save_scene, vmin=ARGS.vmin, vmax=ARGS.vmax, ptp_threshold=ARGS.ptp_threshold)
    elif(INP == 'm'):
        visualizeMesh(mesh)
    elif(inp[0] == 'c'):
        INP = INP.split()
        arrays = getDataArrays(INP[1:])
        
        # masks where labels disagree
        if ARGS.multiclass_labels:
            D = arrays[0]
            P = arrays[1]
            ind = (D != P)
            D[ind] = 7
            visualizeMesh(mesh, [D], color_map=ARGS.color_map, smooth=ARGS.smooth, int_color_map=multiclass_colors, save=ARGS.save_scene, vmin=ARGS.vmin, vmax=ARGS.vmax, ptp_threshold=ARGS.ptp_threshold)
        else:
            D = arrays[0]
            P = arrays[1]
            m_np = (D == 0)*(D != P)
            m_pn = (D == 1)*(D != P)
            D[m_np] = 2
            D[m_pn] = 3
            visualizeMesh(mesh, [D], color_map=ARGS.color_map, smooth=ARGS.smooth, int_color_map=binary_colors, save=ARGS.save_scene)
    else:
        INP = INP.split()
        arrays = getDataArrays(INP)
        
        scene_name = "_".join(INP)
        if ARGS.multiclass_labels:
            visualizeMesh(mesh, arrays, color_map=ARGS.color_map, smooth=ARGS.smooth, int_color_map=multiclass_colors, save=ARGS.save_scene, scene_name=scene_name, vmin=ARGS.vmin, vmax=ARGS.vmax, ptp_threshold=ARGS.ptp_threshold)
        else:
            visualizeMesh(mesh, arrays, color_map=ARGS.color_map, smooth=ARGS.smooth, int_color_map=binary_colors, save=ARGS.save_scene, scene_name=scene_name, vmin=ARGS.vmin, vmax=ARGS.vmax, ptp_threshold=ARGS.ptp_threshold)
