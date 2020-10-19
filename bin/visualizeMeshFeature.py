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
ARGS = PARSER.parse_args()

# builtin modules
import os
import re

# third party modules
import numpy as np
import trimesh

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
    [1.00, 0.00, 0.00], #  4: red
    [0.73, 0.33, 0.83], #  5: purple
    [51/255, 51/255, 1.0] # 6: blue
])

def visualizeMesh(mesh, data=None, colors=None, color_map='seismic', int_color_map=None, max_width=4, shift_axis='x', **kwargs):
    # figure out where to get color information
    if(data is None and colors is None):
        # just visualize the mesh geometry
        return mesh.show(**kwargs)
    elif(colors is None):
        # compute colors from the data array
        vertex_colors = []
        for i in range(len(data)):
            if(data[i].dtype == np.int64 or data[i].dtype == np.bool):
                vertex_colors.append(trimesh.visual.to_rgba(int_color_map[data[i]+1]))
            else:
                vertex_colors.append(trimesh.visual.interpolate(data[i], color_map=color_map))
    else:
        # use the given colors
        if(isinstance(colors, list)):
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
        offset += m.bounding_box.extents[si]
        scene.append(m)
    print("returning scene")
    return trimesh.Scene(scene).show(**kwargs)

def getDataArrays(INP):
    arrays = []
    for inp in INP:
        m = re.match("(w+)(d*)", inp)
        
        if inp in data:
            arrays.append(data[inp])
        elif inp[0] == 'X':
            i = int(inp[1:])
            arrays.append(data['X'][:,i])
        elif inp in extras:
            arrays.append(extras[inp])
        elif inp[0] in extras:
            i = int(inp[1:])
            arrays.append(extras[inp[0]][:,i])
    
    return arrays

# Read in data files
data = np.load(ARGS.data_file)
if ARGS.extras_file:
    extras = np.load(ARGS.extras_file)

if ARGS.point_cloud:
    mesh = trimesh.PointCloud(vertices=data['V'], process=False)
else:
    mesh = trimesh.Trimesh(vertices=data['V'], faces=data['F'], process=False)

# list of features in data file
feature_list = ""
if "feature_names" in data:
    for i, feature in enumerate(data['feature_names']):
        feature_list += "X{:<2d}: {}\n".format(i, feature)
else:
    if 'X' in data:
        for i in range(data['X'].shape[1]):
            feature_list += "X{:<2d}: X{:<2d}\n".format(i, i)
if 'Y' in data:
    feature_list += "Y: ground-truth labels\n"
if ARGS.extras_file:
    for key in extras:
        if extras[key].ndim > 1:
            for i in range(extras[key].shape[1]):
                feature_list += "{}{:<2d}: supplied data field\n".format(key, i)
        else:
            feature_list += "{}: supplied data field\n".format(key)
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
    c to compare two data fields
    q to exit
    a list of data fields, separated by space (e.g. "X1 X2 Y")
:""".strip()

### TODO: add selection like "Y X0 X1 YPR"

# loop input parser
#parser = argparse.ArgumentParser()
#parser.add_argument("--quit", action='store_true', help="exit program")
#parser.add_argument("--list", action='store_true', help="list features available in data file")
#parser.add_argument("-u", type=float, default=None)
#parser.add_argument("-l", type=float, default=None)
#parser.add_argument("x", nargs='+')

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
    elif(INP == 'm'):
        visualizeMesh(mesh)
    elif(inp[0] == 'c'):
        INP = INP.split()
        arrays = getDataArrays(INP[1:])
        
        # masks where labels disagree
        if ARGS.multiclass_labels:
            pass
        else:
            D = arrays[0]
            P = arrays[1]
            m_np = (D == 0)*(D != P)
            m_pn = (D == 1)*(D != P)
            D[m_np] = 2
            D[m_pn] = 3
            visualizeMesh(mesh, [D], color_map=ARGS.color_map, smooth=ARGS.smooth, int_color_map=binary_colors)
    #elif(INP in data or INP in extras):
        #if(INP in data):
            #D = data[INP]
        #else:
            #D = extras[INP]
        #rgb = trimesh.visual.to_rgba(colors[D+1])
        #visualizeMesh(mesh, colors=rgb)
    else:
        INP = INP.split()
        arrays = getDataArrays(INP)
        if ARGS.multiclass_labels:
            visualizeMesh(mesh, arrays, color_map=ARGS.color_map, smooth=ARGS.smooth, int_color_map=multiclass_colors)
        else:
            visualizeMesh(mesh, arrays, color_map=ARGS.color_map, smooth=ARGS.smooth, int_color_map=binary_colors)
    #else:
        #args = parser.parse_args(inp.split())
        #drange = [args.l, args.u]
        #for x in args.x:
            ## check x is in range
            #if(x >= size or x < 0):
                #raise IndexError("{} is out of range [{}, {}]".format(x, 0, size-1))
            #print("Min: {:<.4f} Max: {:<.4f} Avg: {:<.4f} Std: {:<.4f}".format(data['X'][:,x].min(), data['X'][:,x].max(), data['X'][:,x].mean(), data['X'][:,x].std()))
        #visualizeMesh(mesh, data['X'][:, args.x], color_map=ARGS.color_map)
