import numpy as np

class Scaler(object):
    def __init__(self, data, data_range=[None,None]):
        if(data_range[0] is None):
            self.min = data.min()
        else:
            self.min = data_range[0]
        
        if(data_range[1] is None):
            self.max = data.max()
        else:
            self.max = data_range[1]
    
    def __call__(self, value):
        return np.maximum(np.minimum((value-self.min)/(self.max-self.min), 1.0), 0.0)

def writeOFF(file_prefix, vertexs, faces, data=None, colorby="vertex", data_range=[None, None], cmap=None):
    # save mesh as a .OFF file - 
    OUT = open("{}.off".format(file_prefix), "w")
    nv = vertexs.shape[0]
    nf = faces.shape[0]
    
    if(data is not None):
        # get vertex colors
        if(cmap is None):
            cmap = cm.get_cmap('seismic')
            scale = Scaler(data, data_range=data_range)
            rgba = cmap(scale(data))
        else:
            rgba = np.array(list(map(lambda d: cmap(d), data)))
    
    # write vertexes
    if(data is not None and colorby == 'vertex'):
        OUT.write("COFF\n{} {} 0".format(nv, nf))
        for i in range(nv):
            OUT.write("\n{:.6f} {:.6f} {:.6f} {:>.5f} {:>.5f} {:>.5f} 1.0".format(vertexs[i][0], vertexs[i][1], vertexs[i][2], rgba[i][0], rgba[i][1], rgba[i][2]))
    else:
        OUT.write("OFF\n{} {} 0".format(nv, nf))
        for i in range(nv):
            OUT.write("\n{:.6f} {:.6f} {:.6f}".format(vertexs[i][0], vertexs[i][1], vertexs[i][2]))
    
    # write faces
    if(data is not None and colorby == 'face'):
        for i in range(nf):
            #data_value = (data[faces[i][0]] + data[faces[i][1]] + data[faces[i][2]])/3.0
            #rgba = cmap(scale(data_value))
            color = (rgba[faces[i][0]] + rgba[faces[i][1]] + rgba[faces[i][2]])/3.0
            OUT.write("\n{:<4d} {:>5d} {:>5d} {:>5d} {:>.5f} {:>.5f} {:>.5f} 1.0".format(3, faces[i][0], faces[i][1], faces[i][2], color[0], color[1], color[2]))
    else:
        # write faces
        for i in range(nf):
            OUT.write("\n{:<4d} {:>5d} {:>5d} {:>5d}".format(3, faces[i][0], faces[i][1], faces[i][2]))

    OUT.close()

def readOFF(file_name):
    FH = open(file_name)
    # Skip first line
    FH.readline()
    
    # Get number of vertices and faces
    Nv = None
    Nf = None
    while(True):
        line = FH.readline().strip()
        if(len(line) == 0):
            continue
        if(line[0] == '#'):
            continue
        else:
            Nv, Nf, _ = line.split()
            Nv = int(Nv)
            Nf = int(Nf)
            break
    
    # Read in vertexs
    i = 0
    verts = []
    while(i < Nv):
        line = FH.readline().strip()
        verts.append([float(_) for _ in line.split()])
        i += 1
    
    # Read in faces
    i = 0
    faces = []
    while(i < Nf):
        line = FH.readline().strip()
        faces.append([float(_) for _ in line.split()[1:4]])
        i += 1
    
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

def readPLY(file_name):
    pass

def writePLY(file_prefix, vertexs, faces, data=None):
    pass
