#!/usr/bin/env python

# Command-line arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("structures_file",
                help="a list of interface structures to process.")
arg_parser.add_argument("--residue_ids",
                help="a list of residue identifiers that define a binding site")
arg_parser.add_argument("-o", "--output_file", default="processed_files.dat",
                help="Name of output file to write a list of datafiles which were generated.")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="A file storing configuration options in JSON format.")
arg_parser.add_argument("-m", "--moieties_file", dest="moieties_file",
                help="A file in JSON format containing regexes which assign atoms to a class or the name of a built-in label set.")
arg_parser.add_argument("-r", "--refresh", action='store_true', default=False,
                help="recompute mesh/features/labels even if present")
arg_parser.add_argument("-E", "--no_electrostatics", action='store_true',
                help="do not calculate electrostatic features")
arg_parser.add_argument("-L", "--no_labels", dest="no_labels", action='store_true', default=False,
                help="do not generate labels for each interface")
arg_parser.add_argument("-F", "--no_features", dest="no_features", action='store_true', default=False,
                help="do not generate any mesh features for each interface")
arg_parser.add_argument("-A", "--no_adjacency", dest="no_adjacency", action='store_true', default=False,
                help="do not write mesh adjacency matrix to file")
arg_parser.add_argument("-P", "--no_pockets", dest="no_pockets", action='store_true', default=False,
                help="do not use any pocket features")
arg_parser.add_argument("-d", "--debug", action='store_true', default=False,
                help="write additional information")
arg_parser.add_argument("-l", "--ligand_list", dest="ligand_list", nargs="+",
                help="a list of ligand identifiers")
arg_parser.add_argument("-y", "--label_set_name",
                help="a name for the label set")
arg_parser.add_argument("--label_method", choices=["structure_structure_distance", "structure_vertex_distance"], default="structure_vertex_distance",
                help="a name for the label set")
arg_parser.add_argument("--mesh_only", action='store_true',
                help="only compute mesh for each structure")
ARGS = arg_parser.parse_args()

# builtin modules
import logging
import subprocess
import json
import os
from os.path import join as ospj
import pathlib

# third party modules
import numpy as np
from scipy.sparse import save_npz

# geobind modules
import geobind
from geobind import mapStructureFeaturesToMesh, AtomToClassMapper, vertexLabelsToResidueLabels
from geobind.structure.data import data as D
from geobind.structure import StructureData, pairsWithinDistance
from geobind.structure import ResidueMutator
from geobind.utils import Interpolator

def getEntities(structure, regexes, atom_mapper=None, mi=0):
    """Docstring"""
    pro = []
    lig = []
    for chain in structure[mi]:
        for residue in chain:
            resname = residue.get_resname()
            if regexes['PROTEIN']['STANDARD_RESIDUES'].search(resname):
                pro.append(residue.get_full_id())
            
            if atom_mapper:
                if atom_mapper.testResidue(residue):
                    lig.append(residue.get_full_id())
            else:
                lig.append(residue.get_full_id())
    
    return structure.slice(structure, pro, '{}_protein'.format(structure.name)), structure.slice(structure, lig, '{}_ligand'.format(structure.name))

def accumulateSESA(res_dict, atoms, key):
    if isinstance(atoms, StructureData):
        atoms = atoms.atom_list
    
    for atom in atoms:
        if atom.xtra[key] <= 0:
            continue
        
        residue = atom.get_parent()
        resn = residue.get_resname()
        if resn in D.standard_residues:
            _, __, cid, rid = residue.get_full_id()
            rid = "{}.{}.{}".format(cid, rid[1], rid[2])
            if rid not in res_dict:
                res_dict[rid] = {
                    'fsesa': 0.0,
                    'csesa': 0.0,
                    'resn': resn
                }
            
            res_dict[rid][key] += atom.xtra[key]

def main():
    ### Load various data files ####################################################################
    res_mutator = ResidueMutator() # can reuse this for multiple structures
    
    # Get standard surface area
    if(C["AREA_MEASURE"] == "sasa"):
        classifier = Radius()
        classifier.initialize()
        standard_area = D.standard_sasa
    else:
        standard_area = D.standard_sesa
    
    # Create Atom Mapper object
    if not ARGS.no_labels:
        if ARGS.moieties_file:
            atom_mapper = AtomToClassMapper(ARGS.moieties_file)
        elif ARGS.ligand_list:
            name = ARGS.ligand_list_name if ARGS.ligand_list_name else "LIGANDS"
            atom_mapper = AtomToClassMapper(ARGS.ligand_list, default=0, name=name)
        elif "MOIETIES" in C:
            atom_mapper = AtomToClassMapper(C["MOIETIES"])
        elif "LIGAND_LIST" in C:
            atom_mapper = AtomToClassMapper(C["LIGAND_LIST"], default=0, name=C.get("LIGAND_SET_NAME", "LIGANDS"))
        elif "MOIETY_LABEL_SET_NAME" in C:
            atom_mapper = AtomToClassMapper(C["MOIETY_LABEL_SET_NAME"])
        elif ARGS.residue_ids:
            atom_mapper = None
        
        if ARGS.label_set_name:
            label_names = ARGS.label_set_name
        else:
            label_names = "Y_{}".format(atom_mapper.name)
    else:
        atom_mapper = None
    
    ### Load the interface file which describes a list of DNA-protein interfaces to process ########
    listFile = ARGS.structures_file
    PROCESSED = open(ARGS.output_file, "w")
    
    if ARGS.residue_ids:
        RIFH = open(ARGS.residue_ids)
    else:
        RIFH = None
    
    # main loop
    for fileName in open(listFile):
        fileName = fileName.strip()
        if fileName[0] == '#':
            # skip commented lines
            if RIFH:
                RIFH.readline()
            continue
        
        ### LOAD STRUCTURE #########################################################################
        protein_id = '.'.join(fileName.split('.')[0:-1]) + '_protein'
        structure = StructureData(fileName, name=protein_id, path=C["PDB_FILES_PATH"])
        protein, lig = getEntities(structure, D.regexes, atom_mapper)
        
        # Clean the protein entity
        protein, pqr = geobind.structure.cleanProtein(protein, res_mutator, hydrogens=C['HYDROGENS'])
        
        ### MESH GENERATION ########################################################################
        # Write a PDB file matching chain
        pdb = protein.save()
        
        # Generate a mesh
        mesh_prefix = "{}_mesh".format(protein_id)
        mesh_kwargs = dict(op_mode='normal', surface_type='skin', skin_parameter=0.45, grid_scale=C.get("GRID_SCALE", 0.8))
        if ARGS.refresh:
            mesh = geobind.mesh.generateMesh(protein, 
                prefix=mesh_prefix,
                basedir=C["MESH_FILES_PATH"],
                clean=(not ARGS.debug),
                **mesh_kwargs
            )
            logging.info("Computed new mesh for: %s", protein_id)
            meshFile = mesh.save(C['MESH_FILES_PATH'], overwrite=True)
        else:
            meshFile = ospj(C['MESH_FILES_PATH'], "{}_mesh.off".format(protein_id))
            if os.path.exists(meshFile):
                # found an existing mesh, load it
                mesh = geobind.mesh.Mesh(meshFile, name=mesh_prefix)
                logging.info("Loaded existing mesh for: %s", protein_id)
            else:
                # couldn't find mesh files, compute from scratch
                mesh = geobind.mesh.generateMesh(protein, 
                    prefix=mesh_prefix,
                    basedir=C["MESH_FILES_PATH"],
                    clean=(not ARGS.debug),
                    **mesh_kwargs
                )
                logging.info("Computed new mesh for: %s", protein_id)
                meshFile = mesh.save(C['MESH_FILES_PATH'])
                ARGS.refresh = True
        
        ### DATA ARRAYS ############################################################################
        # Check to see what already exists and what we need to compute
        fname = ospj(C['FEATURE_DATA_PATH'], "{}_data.npz".format(protein_id))
        if os.path.exists(fname) and not ARGS.refresh:
            arrays = dict(np.load(fname, allow_pickle=True))
        else:
            arrays = {}
        arrays['V'] = mesh.vertices
        arrays['F'] = mesh.faces
        arrays['name'] = protein_id
        if mesh.vertex_normals is not None:
            arrays['N'] = mesh.vertex_normals
        
        ### FEATURES ###############################################################################
        update_features = (not ARGS.no_features) and (ARGS.refresh or ('X' not in arrays))
        if update_features:
            logging.info("Computing a new set of features for %s" , protein_id)
            FEATURES = []
            FEATURE_NAMES = []
            
            ### Geometry Features ##################################################################
            # Compute vertex-level features based only on mesh geometry. Features are stored in the
            # `mesh.vertex_attributes` property of the mesh object.
            #geobind.mesh.getPatchZernikeMoments(mesh, radius=3.0, order=5, smooth=False)
            geobind.mesh.getMeshCurvature(mesh)
            geobind.mesh.getConvexHullDistance(mesh)
            geobind.mesh.getHKS(mesh)
            
            mk, mv = zip(*mesh.vertex_attributes.items())
            FEATURE_NAMES += list(mk)
            FEATURES += list(mv)
            
            ### Structure Features #################################################################
            # Compute atom-level features and store them in atom.xtra of chain. Atom-level features 
            # are computed based only on the protein structure and are independent of the mesh.

            if C["AREA_MEASURE"] == "sesa":
                geobind.structure.getAtomSESA(protein, protein_id)
            else:
                geobind.structure.getAtomSASA(protein, classifier=classifier)
            
            features_a = [] # store feature names
            features_a += geobind.structure.getSAP(protein, standard_area=standard_area, area_key=C["AREA_MEASURE"], distance=5.0, hydrogens=False)
            features_a += geobind.structure.getCV(protein, 10.00, feature_name="cv_fine", hydrogens=False)
            features_a += geobind.structure.getCV(protein, 25.00, feature_name="cv_medium", hydrogens=False)
            features_a += geobind.structure.getCV(protein, 100.0, feature_name="cv_coarse", hydrogens=False)
            features_a += geobind.structure.getDSSP(protein, pdb)
            features_a += geobind.structure.getAchtleyFactors(protein)
            FEATURE_NAMES += features_a
            
            # Map atom-level features to the mesh, weighted by inverse distance from the atom to 
            # nearby mesh vertices
            Xa = mapStructureFeaturesToMesh(mesh, protein, features_a, distance_cutoff=2.5, weight_method="linear", hydrogens=C["HYDROGENS"])
            FEATURES.append(Xa)
            
            # Map hydrogen bond potenial with binary weights
            features_h = geobind.structure.getHBondAtoms(protein)
            FEATURE_NAMES += features_h
            FEATURES.append( mapStructureFeaturesToMesh(mesh, protein, features_h[0], distance_cutoff=0.5, hydrogens=C["HYDROGENS"], weight_method="linear", impute=False, laplace_smooth=True) )
            FEATURES.append( mapStructureFeaturesToMesh(mesh, protein, features_h[1], distance_cutoff=0.5, hydrogens=C["HYDROGENS"], weight_method="linear", impute=False, laplace_smooth=True) )
            
            # Compute pocket features
            if not ARGS.no_pockets:
                Xp, features_p = geobind.mesh.getPockets(protein, mesh, radius_big=3.0)
                FEATURE_NAMES += features_p
                FEATURES.append(Xp)
            
            # Compute Electrostatic features
            if not ARGS.no_electrostatics:
                # get the potential files
                potfile = ospj(C["ELECTROSTATICS_PATH"], protein_id+"_potential.dx")
                accessfile = ospj(C["ELECTROSTATICS_PATH"], protein_id+"_access.dx")
                if os.path.exists(potfile) and os.path.exists(accessfile):
                    phi = Interpolator(potfile)
                    logging.info("Loaded potential %s from file.", potfile)
                    acc = Interpolator(accessfile)
                    logging.info("Loaded accessibility %s from file.", accessfile)
                else:
                    try:
                        phi, acc = geobind.structure.runAPBS(protein, protein_id, pqr=pqr, basedir=C["ELECTROSTATICS_PATH"])
                    except subpro.CalledProcessError:
                        # try with coarser values
                        phi, acc = geobind.structure.runAPBS(protein, protein_id, pqr=pqr, basedir=C["ELECTROSTATICS_PATH"], space=0.5, cfac=1.5)
                
                Xe, features_e = geobind.mesh.mapElectrostaticPotentialToMesh(mesh, phi, acc, efield=True, diff_method='five_point_stencil', sphere_average=True, laplace_smooth=True)
                FEATURE_NAMES += features_e
                FEATURES.append(Xe)
            
        ### LABELS #############################################################################
        update_labels = (not ARGS.no_labels) and (ARGS.refresh or (label_names not in arrays))
        if update_labels:
            # Compute labels
            logging.info("Computing a new set of labels for %s" , protein_id)
            
            # determine how we define binding site
            if ARGS.residue_ids:
                res_ids = RIFH.readline().strip().split(',')
            elif ARGS.label_method == "structure_vertex_distance":
                res_ids = None
            elif ARGS.label_method == "structure_structure_distance":
                res_ids, lig_ids = pairsWithinDistance(protein, lig, distance=C.get("ATOM_DISTANCE_CUTOFF", 4.0), hydrogens=False)
            
            if res_ids:
                Y = geobind.assignMeshLabelsFromList(protein, mesh, res_ids,
                    weight_method='linear',
                    distance_cutoff=1.5,
                    smooth=C["SMOOTH_LABELS"],
                    smoothing_threshold=C.get("SMOOTHING_THRESHOLD", 50.0),
                    no_smooth=C.get("NO_SMOOTH_CLASSES", [1]),
                    mask=C["MASK_LABELS"],
                    mask_cutoff=C.get("MASK_DISTANCE_CUTOFF", 3.0)
                )
            else:
                Y = geobind.assignMeshLabelsFromStructure(lig, mesh, atom_mapper,
                    smooth=C["SMOOTH_LABELS"],
                    smoothing_threshold=C.get("SMOOTHING_THRESHOLD", 50.0),
                    no_smooth=C.get("NO_SMOOTH_CLASSES", [1]),
                    mask=C["MASK_LABELS"],
                    distance_cutoff=C["MESH_DISTANCE_CUTOFF"],
                    mask_cutoff=C.get("MASK_DISTANCE_CUTOFF", 3.0)
                )
            res_ids = vertexLabelsToResidueLabels(protein, mesh, Y, nc=2)
        
        ### OUTPUT #############################################################################
        # Write features to disk
        
        # Vertex features
        if update_features:
            FEATURES = [x.reshape(-1, 1) if x.ndim == 1 else x for x in FEATURES]
            arrays['X'] = np.concatenate(FEATURES, axis=1)
            arrays['feature_names'] = np.array(FEATURE_NAMES)
        
        # Mesh labels
        if update_labels:
            arrays[label_names] = Y
            if atom_mapper:
                arrays['{}_classes'.format(label_names)] = atom_mapper.classes
            if res_ids:
                arrays['residue_identifiers'] = res_ids
        
        # Write mesh data to disk
        np.savez_compressed(fname, **arrays)
        logging.info("Saved mesh data to disk: %s", fname)
        PROCESSED.write("{}_data.npz\n".format(protein_id))
        
        # Write mesh adjacency to disk
        if not ARGS.no_adjacency:
            fname = ospj(C['FEATURE_DATA_PATH'], "{}_adj.npz".format(protein_id))
            save_npz(fname, mesh.vertex_adjacency_matrix)
            logging.info("Saved adjacency data to disk: %s", fname)
        
        ### CLEAN-UP ###########################################################################
        os.remove(pdb)
        os.remove(pqr)
    
    PROCESSED.close()
    return 0

### Load the config file
with open(ARGS.config_file) as FH:
    C = json.load(FH)
file_path = pathlib.Path(__file__)
C['ROOT_DIR'] = file_path.parent.parent

### Set up logging
log_level = logging.INFO
log_format = '%(levelname)s:    %(message)s'
logging.basicConfig(format=log_format, filename='run.log', level=log_level)

console = logging.StreamHandler()
console.setLevel(log_level)
formatter = logging.Formatter(log_format)
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    main()
