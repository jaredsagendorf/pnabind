# standard packages
import os
import shutil
import subprocess

# third party packages
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO

# geobind packages
from geobind.mesh import Mesh

def __move(fileName, dest):
    path = os.path.join(dest, fileName)
    if(os.path.exists(path)):
        if(os.path.abspath(dest) != os.getcwd()):
            os.remove(path)
            shutil.move(fileName, dest)
    else:
        shutil.move(fileName, dest)

def runMSMS(atom_list, file_prefix, basedir, 
        clean=True, quiet=True, hydrogens=True, area_only=False, **kwargs
    ):    
    # generate coordinate file
    coordFile = "{}_coords.xyzr".format(file_prefix)
    FH = open(coordFile, "w")
    for atom in atom_list:
        if((not hydrogens) and atom.element == "H"):
            continue
        atmn = atom.name
        acoords = atom.get_coord()
        radius = atom.xtra["radius"]
        FH.write("{:7.4f} {:7.4f} {:7.4f} {:3.2f}\n".format(acoords[0], acoords[1], acoords[2], radius))
    FH.close()
    
    # set MSMS options
    msms_opts = {
        'probe_radius': 1.5,
        'density': 1.0,
        'hdensity': 3.0,
        'surface': 'tses'
    }
    msms_opts.update(kwargs)
    if(area_only):
        msms_opts['surface'] = 'ases'
    
    # run MSMS and generate vertex and face file
    args = [
        "msms",
        "-probe_radius", str(msms_opts['probe_radius']),
        "-density", str(msms_opts['density']),
        "-hdensity", str(msms_opts['hdensity']),
        "-if", coordFile,
        "-of", file_prefix,
        "-af", "{}.area".format(file_prefix),
        "-surface",  msms_opts["surface"]
    ]
    if(quiet):
        FNULL = open(os.devnull, 'w')
        subprocess.call(args, stdout=FNULL, stderr=FNULL)
        FNULL.close()
    else:
        subprocess.call(args)
    
    if(area_only):
        # delete/move files and return path to the area file
        if(clean):
            os.remove("{}_coords.xyzr".format(file_prefix))
        else:
            __move("{}_coords.xyzr".format(file_prefix), basedir)
        af = "{}.area".format(file_prefix)
        __move(af, basedir)
        
        return os.path.join(os.path.abspath(basedir), af)
    else:
        # Get vertices
        vertData = open("{}.vert".format(file_prefix)).readlines()[2:]
        vertexs = []
        normals = []
        for line in vertData[1:]:
            line = line.strip().split()
            vertex = np.array(line[0:3], dtype=np.float32)
            normal = np.array(line[3:6], dtype=np.float32)
            
            vertexs.append(vertex)
            normals.append(normal)
        vertexs = np.array(vertexs, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        
        # Get faces
        faceData = open("{}.face".format(file_prefix)).readlines()[3:]
        faces = []
        for line in faceData:
            line = line.strip().split()
            i = int(line[0])-1
            j = int(line[1])-1
            k = int(line[2])-1
            faces.append([i,j,k])
        faces = np.array(faces, dtype=np.int32)
        
        # clean up MSMS files
        if(clean):
            os.remove("{}.face".format(file_prefix))
            os.remove("{}.vert".format(file_prefix))
            os.remove("{}_coords.xyzr".format(file_prefix))
            os.remove("{}.area".format(file_prefix))
        else:
            __move("{}.face".format(file_prefix), basedir)
            __move("{}.vert".format(file_prefix), basedir)
            __move("{}_coords.xyzr".format(file_prefix), basedir)
            __move("{}.area".format(file_prefix), basedir)
        
        # return mesh
        return Mesh(V=vertexs, F=faces, N=normals, name=file_prefix)

def runNanoShaper(atom_list, file_prefix, basedir, 
        clean=True, quiet=True, hydrogens=True, pockets_only=False, **kwargs
    ):
    # generate coordinate file
    coordFile = "{}.xyzr".format(file_prefix)
    FH = open(coordFile, "w")
    for atom in atom_list:
        if((not hydrogens) and atom.element == "H"):
            continue
        atmn = atom.name
        acoords = atom.get_coord()
        radius = atom.xtra["radius"]
        FH.write("{:7.4f} {:7.4f} {:7.4f} {:3.2f} {}\n".format(acoords[0], acoords[1], acoords[2], radius, atom.serial_number))
    FH.close()
    
    # run NanoShaper and generate .OFF file
    nanoshaper_args = {
        "grid_scale": 2.0,
        "grid_perfil": 90.0,
        "op_mode": "normal",
        "surface_type": "skin",
        "build_status_map": "true",
        "cavity_filling_volume": 100,
        "skin_parameter": 0.45,
        "accurate_triangulation": "true",
        "smooth_mesh": "true",
        "blobbyness": -2.5,
        "radius_big": 3.0
    }
    nanoshaper_args.update(kwargs)
    if(pockets_only):
        # change operation mode
        nanoshaper_args['op_mode'] = 'pockets'
    prm_template = """
# Global Parameters
Operative_Mode = {op_mode}
Grid_scale = {grid_scale}
Grid_perfil = {grid_perfil}
Number_thread = 32 
# Map Settings
Build_epsilon_maps = false
Build_status_map = {build_status_map}
# Surface Parameters
Surface = {surface_type}
Skin_Surface_Parameter = {skin_parameter}
Blobbyness = {blobbyness}
Skin_Fast_Projection = false
Accurate_Triangulation = {accurate_triangulation}
Triangulation = true
Check_duplicated_vertices = true
Smooth_Mesh = {smooth_mesh}
# Pocket/Cavities settings
Pockets_And_Cavities = false
Cavity_Detection_Filling = true
Keep_Water_Shaped_Cavities = false
Conditional_Volume_Filling_Value = {cavity_filling_volume}
Num_Wat_Pocket = 2
Pocket_Radius_Big = {radius_big}
Pocket_Radius_Small = 1.4
# I/O Settings
XYZR_FileName = {file_prefix}.xyzr
Save_Cavities = false
Save_Status_map = false
Vertex_Atom_Info = false"""

    PRM = open("{}.prm".format(file_prefix), 'w')
    PRM.write(prm_template.format(file_prefix=file_prefix, **nanoshaper_args))
    PRM.close()
    args = [
        "NanoShaper",
        "{}.prm".format(file_prefix)
    ]
    if(quiet):
        FNULL = open(os.devnull, 'w')
        subprocess.call(args, stdout=FNULL, stderr=FNULL)
        FNULL.close()
    else:
        subprocess.call(args)
    
    if(pockets_only):
        # Do not return a mesh, exit from here
        if(clean):
            os.remove("{}.prm".format(file_prefix))
            os.remove("{}.xyzr".format(file_prefix))
        return
    
    # rename mesh file
    os.rename("triangulatedSurf.off", "{}.off".format(file_prefix))
    
    # clean up NanoShaper files
    if(clean):
        os.remove("{}.prm".format(file_prefix))
        os.remove("{}.xyzr".format(file_prefix))
    else:
        __move("{}.prm".format(file_prefix), basedir)
        __move("{}.xyzr".format(file_prefix), basedir)
    
    # move meshfile to base dir
    meshfile = "{}.off".format(file_prefix)
    if(basedir != os.getcwd()):
        __move(meshfile, basedir)
        meshfile = os.path.join(basedir, meshfile)
    return Mesh(handle=meshfile, name=file_prefix)

def runEDTSurf(pdbFile, file_prefix, basedir, clean=True, quiet=True, **kwargs):
    # run EDTSurf and generate .PLY file
    edtsurf_args = {
        "-t": "2",
        "-s": "3",
        "-c": "1",
        "-p": "1.4",
        "-f": "2.0",
        "-h": "2",
        "-o": file_prefix
    }
    edtsurf_args.update(kwargs)
    
    args = [
        "EDTSurf",
        "-i",
        pdbFile
    ]
    for key in edtsurf_args:
        args.append(key)
        args.append(edtsurf_args[key])
    if(quiet):
        FNULL = open(os.devnull, 'w')
        subprocess.call(args, stdout=FNULL, stderr=FNULL)
        FNULL.close()
    else:
        subprocess.call(args)
    
    # clean up EDTSurf files
    if(clean):
        os.remove("{}-cav.pdb".format(file_prefix))

def generateMesh(structure, 
        prefix=None, basedir=None, clean=True, hydrogens=True, quiet=True, 
        method='nanoshaper', level='S', entity_id=None, kwargs=dict()
    ):
    # Check what we have been given
    if(isinstance(structure, str)):
        # Check if PQR file exists
        if(not os.path.exists(structure)):
            raise ValueError("Can not find PQR file: {}".format(structure))
        # Load a PQR file
        if(prefix is None):
            prefix = ".".join(os.path.basename(structure).split('.')[:-1]) # strip the file extension
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
        structure = parser.get_structure(prefix, structure)
        # Choose which part of the structure we want to use to generate a mesh
        if(level == 'M'):
            structure = structure[entity_id[0]]
        elif(level == 'C'):
            structure = structure[entity_id[0]][entity_id[1]]
        
        # Get atom list
        atom_list = []
        for atom in structure.get_atoms():
            atom.xtra["radius"] = atom.bfactor
            atom.xtra["charge"] = atom.occupancy
            atom_list.append(atom)
    else:
        level = structure.get_level()
        if(prefix is None):
            # Use the structure id as the prefix
            if(level == 'S'):
                prefix = structure.get_id()
            elif(level == 'M'):
                prefix = structure.get_parent().get_id()
            elif(level == 'C'):
                prefix = structure.get_parent().get_parent().get_id()
        
        # Get atom list
        atom_list = []
        for atom in structure.get_atoms():
            atom_list.append(atom)
    
    if(basedir is None):
        basedir = os.getcwd()
    
    if(method == 'nanoshaper'):
        # Run NanoShaper
        mesh = runNanoShaper(atom_list, prefix, basedir, clean=clean, hydrogens=hydrogens, quiet=quiet, **kwargs)
    elif(method == 'msms'):
        # Run MSMS
        mesh = runMSMS(atom_list, prefix, basedir, clean=clean, hydrogens=hydrogens, quiet=quiet, **kwargs)
    
    return mesh
