#!/usr/bin/env python

# command line args
import argparse
PARSER = argparse.ArgumentParser()
PARSER.add_argument("data_file")
PARSER.add_argument("extras_file", nargs='?')
PARSER.add_argument("--color_map", dest='color_map', default='seismic',
        help="Name of a matplotlib color map.")
ARGS = PARSER.parse_args()

# builtin modules
import os

# third party modules
import numpy as np
import trimesh

# geobind modules
def visualizeMesh(mesh, data=None, colors=None, color_map='seismic', max_width=4, shift_axis='x', **kwargs):
    # figure out where to get color information
    if(data is None and colors is None):
        # just visualize the mesh geometry
        return mesh.show(**kwargs)
    elif(colors is None):
        # compute colors from the data array
        vertex_colors = []
        for i in range(data.shape[1]):
            vertex_colors.append(trimesh.visual.interpolate(data[:,i], color_map=color_map))
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

# Read in data files
data = np.load(ARGS.data_file)
if(ARGS.extras_file):
    extras = np.load(ARGS.extras_file)
mesh = trimesh.Trimesh(vertices=data['V'], faces=data['F'], process=False)

# loop input parser
parser = argparse.ArgumentParser()
parser.add_argument("--quit", action='store_true', help="exit program")
parser.add_argument("--list", action='store_true', help="list features available in data file")
parser.add_argument("-u", type=float, default=None)
parser.add_argument("-l", type=float, default=None)
parser.add_argument("x", type=int, nargs='+')

# list of features in data file
if "feature_names" in data:
    feature_list = ""
    for i, feature in enumerate(data['feature_names']):
        feature_list += "{:2d}: {}\n".format(i, feature)
    feauture_list = feature_list.strip()
else:
    feature_list = "no feature data found!"

# feature array
if("X" in data):
    size = data["X"].shape[1]
else:
    size = -1

# binary label colors
colors = np.array([
    [1.00, 1.00, 1.00], # -1: masked
    [0.55, 0.70, 0.40], #  0: non-binding site
    [1.00, 0.50, 0.00], #  1: binding site
    [1.00, 0.00, 0.00], #  2: false positives
    [0.73, 0.33, 0.83], #  3: false negatives
])

# input string
input_string = """
Enter one of the following:
    l to list features
    m to visualize only the mesh
    q to exit
    the single-character name of a data field
    one or more feature indices to visualize
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
    elif(INP == 'm'):
        visualizeMesh(mesh)
    elif(INP in data or INP in extras):
        if(INP in data):
            D = data[INP]
        else:
            D = extras[INP]
        rgb = trimesh.visual.to_rgba(colors[D+1])
        visualizeMesh(mesh, colors=rgb)
    elif(inp.strip() == 'c'):
        pass
        #P = np.load(ARGS.compare_file)
        ## masks where labels disagree
        #m_np = (D == 0)*(D != P)
        #m_pn = (D == 1)*(D != P)
        #D[m_np] = 2
        #D[m_pn] = 3
    else:
        args = parser.parse_args(inp.split())
        drange = [args.l, args.u]
        for x in args.x:
            # check x is in range
            if(x >= size or x < 0):
                raise IndexError("{} is out of range [{}, {}]".format(x, 0, size-1))
            print("Min: {:<.4f} Max: {:<.4f} Avg: {:<.4f} Std: {:<.4f}".format(data['X'][:,x].min(), data['X'][:,x].max(), data['X'][:,x].mean(), data['X'][:,x].std()))
        visualizeMesh(mesh, data['X'][:, args.x], color_map=ARGS.color_map)
