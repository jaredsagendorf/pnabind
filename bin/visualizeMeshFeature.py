import os
import argparse
import sys
import subprocess
import numpy as np
from geobind import writeOFF, readOFF

PARSER = argparse.ArgumentParser()
PARSER.add_argument("mesh_file")
PARSER.add_argument("feature_file")
PARSER.add_argument("compare_file", nargs='?')
PARSER.add_argument("-l", dest='labels', action="store_true", default=False)
ARGS = PARSER.parse_args()

class colorMapper(object):
    def __init__(self, colors):
        self.colors = colors
    def __call__(self, value):
        return self.colors[value]

#meshFile = os.path.join(ARGS.mesh_dir, ARGS.prefix+"_mesh.off")
#if(ARGS.prediction_dir):
    #predFile =  os.path.join(ARGS.prediction_dir, ARGS.prefix+"_vertex_labels_p.npy")
    #ARGS.labels = True
    #P = np.load(predFile)

#if(ARGS.labels):
    #dataFile = os.path.join(ARGS.feature_dir, ARGS.prefix+"_vertex_labels.npy")
    #if(not os.path.exists(dataFile)):
        #dataFile = os.path.join(ARGS.feature_dir, ARGS.prefix+"_vertex_labels_p.npy")
#else:
    #dataFile = os.path.join(ARGS.feature_dir, ARGS.prefix+"_vertex_features.npy")


V, F = readOFF(ARGS.mesh_file)
D = np.load(ARGS.feature_file)

parser = argparse.ArgumentParser()
parser.add_argument("-u", type=float, default=None)
parser.add_argument("-l", type=float, default=None)
parser.add_argument("x", type=int, nargs='+')

if(not ARGS.labels):
    size = D.shape[1]
    while True:
        inp = raw_input("Enter an integer index [{}, {}] into the data to visualize (q to exit): ".format(0, size-1)).strip()
        
        # check if we are quitting
        if(inp == 'q'):
            break
        else:
            args = parser.parse_args(inp.split())
        
        drange = [args.l, args.u]
        fnames = ["geomview"]
        for x in args.x:
            # check x is in range
            if(x >= size or x < 0):
                print("{} is out of range [{}, {}]".format(x, 0, size-1))
                continue
            
            print("Min: {:<.4f} Max: {:<.4f} Avg: {:<.4f} Std: {:<.4f}".format(D[:,x].min(), D[:,x].max(), D[:,x].mean(), D[:,x].std()))
            handle = "temp_off_{}".format(x)
            writeOFF(handle, V, F, D[:,x], colorby='face', data_range=drange)
            fnames.append("{}.off".format(handle))
        subprocess.call(fnames)
        for f in fnames[1:]:
            os.remove(f)
else:
    if(ARGS.compare_file):
        P = np.load(ARGS.compare_file)
        # masks where labels disagree
        m_np = (D == 0)*(D != P)
        m_pn = (D == 1)*(D != P)
        D[m_np] = 2
        D[m_pn] = 3

    cm = colorMapper({
        0:  (0.55, 0.70, 0.40), #(0.5, 0.0, 0.0),
        1:  (1.00, 0.50, 0.00), #(0.0, 0.0, 0.3),
        -1: (1.0, 1.0, 1.0), 
        2:  (1.0, 0.0, 0.0), # false positives (red)
        3:  (0.73, 0.33, 0.83)  # false negatives (orchard)
    })
    
    handle = "temp_off"
    writeOFF(handle, V, F, data=D, colorby='vertex', cmap=cm)
    fname = "{}.off".format(handle)
    subprocess.call(["geomview", fname])
    os.remove(fname)
