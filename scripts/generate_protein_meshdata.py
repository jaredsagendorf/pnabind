#!/usr/bin/env python

# Command-line arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("structures_file",
                help="a list of protein structures to process")
arg_parser.add_argument("--residue_ids",
                help="a list of residue identifiers that define a binding site")
arg_parser.add_argument("-o", "--output_file", default=None,
                help="output file to write a list of datafiles which were generated")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="a file storing configuration options in JSON format")
arg_parser.add_argument("-m", "--moieties_file", dest="moieties_file",
                help="a file in JSON format containing regexes which assign atoms to a class or the name of a built-in label set")
arg_parser.add_argument("-r", "--refresh", action='store_true', default=False,
                help="recompute mesh/features/labels even if present in an existing datafile")
arg_parser.add_argument("-E", "--no_electrostatics", dest="electrostatics", action='store_false', default=True,
                help="do not calculate electrostatic features")
arg_parser.add_argument("-Y", "--no_labels", dest="labels", action='store_false', default=True,
                help="do not generate binding-site labels")
arg_parser.add_argument("-F", "--no_features", dest="features", action='store_false', default=True,
                help="do not generate any features")
arg_parser.add_argument("-A", "--no_adjacency", dest="adjacency", action='store_false', default=True,
                help="do not write mesh adjacency matrix to file")
arg_parser.add_argument("-P", "--no_pockets", dest="pockets", action='store_false', default=True,
                help="do not use any pocket features")
arg_parser.add_argument("-Z", "--no_zernike", dest="zernike", action='store_false', default=True,
                help="do not compute zernike polynomials")
arg_parser.add_argument("-M", "--no_msa", dest="msa", action='store_false', default=True,
                help="do not compute MSA features")
arg_parser.add_argument("-d", "--debug", action='store_true', default=False,
                help="print out additional information")
arg_parser.add_argument("-l", "--ligand_list", dest="ligand_list", nargs="+",
                help="a list of ligand identifiers to be used for binding-site labelling")
arg_parser.add_argument("-y", "--label_key",
                help="a key name for binding labels")
arg_parser.add_argument("--label_method", choices=["structure_structure_distance", "structure_vertex_distance", "none"], default="structure_structure_distance",
                help="method for computing vertex labels.")
arg_parser.add_argument("--mesh_only", action='store_true',
                help="only compute mesh for each structure, no features/labels")
arg_parser.add_argument("--mask_chains",
                help="mask vertices corresponding to each chain in given json file.")
arg_parser.add_argument("--apbs", action='store_true',
                help="prefer APBS over TABI-PB for electrostatic calculations")
arg_parser.add_argument("--nprocs", default=None, type=int,
                help="number of processors to use for parallelization")
arg_parser.add_argument("-s", "--save_structure", action="store_true",
                help="save protein and ligand structures to file")
arg_parser.add_argument("--ignore_los", action="store_false", dest="check_label_los", default=True,
                help="check that binding site labels respect line-of-sight to bound ligand")
arg_parser.add_argument("--full_chain_sequences",
                help="use full chain sequences for MSA features")
arg_parser.add_argument("-a", "--auxiliary", action='store_true',
                help="process auxiliary training data")
arg_parser.add_argument("--prune_structures", action='store_true',
                help="prune large structures")
arg_parser.add_argument("--clean_structures", action='store_true',
                help="fix common issues with protein structures")
ARGS = arg_parser.parse_args()

# builtin modules
import logging
import subprocess
import json
import os
import re
from os.path import join as ospj
import pathlib

# third party modules
import numpy as np
from scipy.sparse import save_npz
from scipy.spatial import cKDTree

# pnabind modules
import pnabind
from pnabind import mapStructureFeaturesToMesh, AtomToClassMapper
from pnabind import vertexLabelsToResidueLabels
from pnabind.structure.data import data as D
from pnabind.structure import StructureData, pairsWithinDistance
from pnabind.structure import ResidueMutator, splitEntities, Radius
from pnabind.structure import getMSAFeatures, getAtomChargeRadius, runPDB2PQR, pruneStructure, ChainSequence
from pnabind.utils import Interpolator, clipOutliers, segmentsIntersectTriangles

def smoothResidues(protein, int_residues, iterations=1, area_threshold=5.0, mesh=None):
    if not ARGS.features and area_threshold > 0:
        pnabind.structure.getAtomSESA(protein, prefix="tmp", mesh=mesh)
    residues = pnabind.structure.aggregateResidueAtoms(protein.get_atoms(), C["AREA_MEASURE"])
    Sres = []
    Yres = []
    for i, r in enumerate(residues):
        if r.xtra[C["AREA_MEASURE"]] > area_threshold:
            if r in int_residues:
                Yres.append(1)
            else:
                Yres.append(0)
            Sres.append(r)
    Yres = pnabind.smoothResidueLabels(Sres, np.array(Yres), ignore_class=[1], iterations=iterations)
    for i, r in enumerate(Sres):
        if Yres[i] == 1:
            int_residues.add(r)

def checkMeshIntersection(mesh, Y, atoms, distance_cutoff=5.0):    
    # get KDTree of query atoms
    Xa = np.array([a.coord for a in filter(lambda x: x.element != 'H', atoms)])
    if len(Xa) == 0:
        return Y
    
    kdt = cKDTree(Xa)
    
    # get binding vertices
    Xv = mesh.vertices[Y==1]
    Nv = len(Xv)
    
    # loop over segments and check if each one intersects mesh
    V = mesh.vertices
    F = mesh.faces
    cmask = np.zeros(Nv, dtype=bool)
    for i in range(Nv):
        xi = kdt.query_ball_point(Xv[i], distance_cutoff) # atoms in cutoff distance
        if len(xi) == 0:
            continue
        fi = mesh.facesInBall(Xv[i], 1.0+distance_cutoff, return_indices=True) # face indices centered on vertex
        T = [V[F[fi,0]], V[F[fi,1]], V[F[fi,2]]]
        s = (np.tile(Xv[i], (len(xi), 1)), Xa[xi])
        cmask[i] = np.any(segmentsIntersectTriangles(s, T) == 0)
    
    # update labels
    yi = np.arange(len(Y))
    vi = np.arange(Nv)
    ind = vi[~cmask]
    Y[yi[Y==1][ind]] = 0
    
    return Y

def getResidues(residue_ids, protein, dnaprodb_format=True, chain_map=None):
    residues = []
    if protein.get_level() == "S":
        protein = protein[0]
    
    for rid in residue_ids:
        if dnaprodb_format:
            cid, num, ins = rid.split('.')
            ins = " " if ins == "_" else ins
            rid = (' ', int(num), ins)
        else:
            cid = rid[2]
            rid = rid[3]
        
        if chain_map and (cid in chain_map):
            cid = chain_map[cid]
        
        if cid in protein and rid in protein[cid]:
            residues.append(protein[cid][rid])
        else:
            logging.info("%s.%s: residue '%s' not found for label mapping!", protein.name, cid, rid)
    
    return residues

def makeMesh(structure, prefix):
    mesh_kwargs = dict(
        op_mode='normal',
        surface_type='skin', 
        skin_surface_parameter=0.45,
        grid_scale=C.get("GRID_SCALE", 0.8),
        mesh_kwargs={"remove_disconnected_components": C.get("REMOVE_DISCONNECTED_COMPONENTS", True)}
    )
    mesh = pnabind.mesh.generateMesh(structure, 
        prefix=prefix,
        basedir=C["MESH_FILES_PATH"],
        clean=(not ARGS.debug),
        fallback="msms",
        **mesh_kwargs
    )
    
    return mesh

def main():
    ### Load various data files ####################################################################
    res_mutator = ResidueMutator() # can reuse this for multiple structures
    
    # Get standard surface area
    classifier = Radius()
    classifier.initialize()
    if C["AREA_MEASURE"] == "sasa":
        standard_area = D.standard_sasa
    else:
        standard_area = D.standard_sesa
    
    # Create Atom Mapper object
    if ARGS.labels:
        atom_mapper = None
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
        
        if ARGS.label_key:
            label_key = ARGS.label_key
        else:
            label_key = "Y_{}".format(atom_mapper.name)
    else:
        atom_mapper = None
    
    # Check for masked chains file
    MASKED_CHAINS_MAP = None
    if ARGS.mask_chains:
        with open(ARGS.mask_chains) as FH:
            MASKED_CHAINS_MAP = json.load(FH)
    
    # Check for residue id file
    BINDING_RESIDUES_MAP = None
    if ARGS.residue_ids:
        with open(ARGS.residue_ids) as FH:
            BINDING_RESIDUES_MAP = json.load(FH)
    
    # Check for chain sequence file
    FULL_CHAIN_SEQUENCES = None
    if ARGS.full_chain_sequences:
        with open(ARGS.full_chain_sequences) as FH:
            FULL_CHAIN_SEQUENCES = json.load(FH)
    
    # Output for successfully processed file
    if ARGS.output_file is not None:
        PROCESSED = open(ARGS.output_file, "w")
    
    ### Load the interface file which describes a list of DNA-protein interfaces to process ########
    listFile = ARGS.structures_file
    
    # main loop
    FNULL = open(os.devnull, 'w')
    for fileName in open(listFile):
        fileName = fileName.strip()
        if fileName[0] == '#':
            # skip commented lines
            continue
        
        ### LOAD STRUCTURE #########################################################################
        structure_id = '.'.join(fileName.split('.')[0:-1]) # strip off file extension
        pfile = ospj(C['STRUCTURES_OUTPUT_PATH'], "%s_protein.pdb" % structure_id)
        lfile = ospj(C['STRUCTURES_OUTPUT_PATH'], "%s_ligand.pdb" % structure_id)
        if not ARGS.refresh and os.path.exists(pfile) and os.path.exists(lfile):
            protein_id = structure_id + "_protein"
            protein = StructureData("%s.pdb" % protein_id, name=protein_id, path=C['STRUCTURES_OUTPUT_PATH'], level='M')
            lig = StructureData("%s_ligand.pdb" % structure_id, name="ligand", path=C['STRUCTURES_OUTPUT_PATH'])
            logging.info("Loaded saved protein and ligand structures for %s" , protein_id)
        else:
            structure = StructureData(fileName, name=structure_id, path=C["PDB_FILES_PATH"])
            if atom_mapper is not None:
                protein, lig = splitEntities(structure, D.regexes, atom_mapper, min_chain_length=20)
            else:
                protein = structure
                protein.name = protein.name + "_protein"
                lig = None
            protein_id = protein.name
            
            # Clean the protein entity
            if ARGS.clean_structures:
                protein = pnabind.structure.cleanProtein(protein, res_mutator, remove_hydrogens=(not C['HYDROGENS']), method="pdbfixer")
            chain_sequence_map = {}
            if ARGS.prune_structures:
                target_chains = set([_[0] for _ in BINDING_RESIDUES_MAP[fileName]])
                for chain in protein:
                    cid = chain.get_id()
                    if FULL_CHAIN_SEQUENCES:
                        full_seq = FULL_CHAIN_SEQUENCES[fileName][cid]
                    else:
                        full_seq = None
                    chain_sequence_map[cid] = ChainSequence(chain, full_seq=full_seq)
                pruneStructure(protein, target_chains, size_limit=C.get("PRUNE_STRUCTURE_SIZE_LIMIT", 25000))
            if ARGS.save_structure:
                # Write cleaned protein and ligand to disk
                save_path = ospj(C.get('STRUCTURES_OUTPUT_PATH', '.'), "%s.pdb")
                protein.save(save_path % protein.name)
                if lig:
                    lig.save(save_path % lig.name)
        getAtomChargeRadius(protein)
        
        ### DATA ARRAYS ############################################################################
        # Check to see what already exists and what we need to compute
        fname = ospj(C['FEATURE_DATA_PATH'], "{}_data.npz".format(protein_id))
        
        if os.path.exists(fname) and not ARGS.refresh:
            # load existing data file
            arrays = dict(np.load(fname, allow_pickle=True))
            mesh = pnabind.mesh.Mesh(vertices=arrays['V'], faces=arrays['F'], remove_disconnected_components=False)
        else:
            # compute everything from scratch
            arrays = {}
            arrays['name'] = protein_id
            
            # generate a mesh
            mesh_prefix = "{}_mesh".format(protein_id)
            mesh = makeMesh(protein, mesh_prefix)
            mesh.save(C['MESH_FILES_PATH'], overwrite=True)
            logging.info("Computed new mesh for: %s", protein_id)
            
            arrays['V'] = mesh.vertices
            arrays['F'] = mesh.faces
            if mesh.vertex_normals is not None:
                arrays['N'] = mesh.vertex_normals
        
        ### FEATURES ###############################################################################
        if ARGS.features:
            logging.info("Computing a new set of features for %s" , protein_id)
            FEATURES = []
            FEATURE_NAMES = []
            
            ### MSA Features ######################################################################
            if ARGS.msa:
                logging.info("Running PSIBLAST and HHBLITS for each chain in %s" , protein_id)
                
                # Iterate over each protein chain
                for chain in protein.get_chains():
                    cid = chain.get_id()
                    if cid in chain_sequence_map:
                        seq_map = chain_sequence_map[cid]
                    else:
                        seq_map = None
                        if FULL_CHAIN_SEQUENCES:
                            full_seq = FULL_CHAIN_SEQUENCES[fileName][cid]
                        else:
                            full_seq = None
                    features_s = getMSAFeatures(chain,
                        seq_map=seq_map,
                        full_seq=full_seq,
                        blits_db_name=C['HHBLITS_DB_NAME'],
                        blits_db_dir=C['HHBLITS_DB_DIR'],
                        blast_db_name=C["BLAST_DB_NAME"],
                        blast_db_dir=C["BLAST_DB_DIR"],
                        blits_kwargs=dict(cpu=8),
                        blast_kwargs=dict(num_threads=8, word_size=3),
                        run_blits=True,
                        run_blast=True
                    )
                FEATURE_NAMES += features_s
                
                Xs = mapStructureFeaturesToMesh(mesh, protein, features_s, distance_cutoff=2.5, weight_method="linear", include_hydrogens=C["HYDROGENS"])
                FEATURES.append(Xs)
            
            ### Geometry Features ##################################################################
            # Compute vertex-level features based only on mesh geometry. Features are stored in the
            # `mesh.vertex_attributes` property of the mesh object.
            if ARGS.zernike:
                pnabind.mesh.getPatchDescriptors(mesh, radius=17.5, n_max=5, sample_ratio=0.2, nprocs=ARGS.nprocs)
            pnabind.mesh.getMeshCurvature(mesh, gaussian_curvature=False, shape_index=False)
            pnabind.mesh.getHKS(mesh, num_samples=4)
            
            mk, mv = zip(*mesh.vertex_attributes.items())
            FEATURE_NAMES += list(mk)
            FEATURES += list(mv)
            
            ### Structure Features #################################################################
            # Compute atom-level features and store them in atom.xtra of chain. Atom-level features 
            # are computed based only on the protein structure and are independent of the mesh.
            if C["AREA_MEASURE"] == "sesa":
                try:
                    if mesh.num_components > 1:
                        # msms doesn't handle disconnected meshes properly
                        pnabind.structure.getAtomSESA(protein, mesh=mesh)
                    else:
                        pnabind.structure.getAtomSESA(protein, prefix=protein_id)
                except:
                    # rarely msms will crash, this is a safe fall-back
                    pnabind.structure.getAtomSESA(protein, mesh=mesh)
            else:
                pnabind.structure.getAtomSASA(protein, classifier=classifier)
            
            features_a = [] # store feature names
            features_a += [pnabind.structure.getSAP(protein, standard_area=standard_area, area_key=C["AREA_MEASURE"], distance=5.0, impute_hydrogens=True)]
            features_a += [pnabind.structure.getCV(protein, 7.5, feature_name="cv_fine", impute_hydrogens=True)]
            features_a += [pnabind.structure.getCV(protein, 15.0, feature_name="cv_medium", impute_hydrogens=True)]
            features_a += [pnabind.structure.getCV(protein, 30.0, feature_name="cv_coarse", impute_hydrogens=True)]
            features_a += pnabind.structure.getAchtleyFactors(protein)
            FEATURE_NAMES += features_a
            
            # Map atom-level features to the mesh, weighted by inverse distance from the atom to 
            # nearby mesh vertices
            Xa = mapStructureFeaturesToMesh(mesh, protein, features_a, distance_cutoff=2.5, weight_method="linear", include_hydrogens=C["HYDROGENS"])
            FEATURES.append(Xa)
            
            # Map hydrogen bond potenial with binary weights
            features_h = pnabind.structure.getHBondAtoms(protein, feature_value=2.0)
            FEATURE_NAMES += features_h
            FEATURES.append( mapStructureFeaturesToMesh(mesh, protein, features_h, include_hydrogens=C["HYDROGENS"], map_to="nearest", laplace_smooth=True, iterations=5) )
            
            # Compute pocket features
            if ARGS.pockets:
                Xp, features_p = pnabind.mesh.getPockets(protein, mesh, radius_big=3.0)
                FEATURE_NAMES += features_p
                FEATURES.append(Xp)
            
            # Compute Electrostatic features
            if ARGS.electrostatics:
                # check which available on system path
                tabi_avail = not subprocess.call(['which', "tabipb"], stdout=FNULL, stderr=FNULL)
                apbs_avail = not subprocess.call(['which', "apbs"], stdout=FNULL, stderr=FNULL)
                
                pqr = runPDB2PQR(protein, replace_hydrogens=False)
                if ARGS.apbs and apbs_avail:
                    # get the potential files
                    potfile = ospj(C["ELECTROSTATICS_PATH"], protein_id+"_potential.dx")
                    accessfile = ospj(C["ELECTROSTATICS_PATH"], protein_id+"_access.dx")
                    if os.path.exists(potfile) and os.path.exists(accessfile):
                        phi = Interpolator(potfile)
                        logging.info("Loaded potential %s from file.", potfile)
                        acc = Interpolator(accessfile)
                        logging.info("Loaded accessibility %s from file.", accessfile)
                    else:
                        phi, acc = pnabind.structure.runAPBS(pqr, prefix=protein_id, basedir=C["ELECTROSTATICS_PATH"], **C.get("ELECTROSTATICS_KW", {}))
                    Xe, features_e = pnabind.mesh.mapElectrostaticPotentialToMesh(mesh, phi, acc, efield=True, diff_method='five_point_stencil', sphere_average=True, laplace_smooth=True, scale_to_tabi=True)
                    
                    FEATURE_NAMES += features_e
                    FEATURES.append(Xe)
                elif tabi_avail:
                    try:
                        points, faces, normals, phi, nphi = pnabind.structure.runTABIPB(pqr, prefix=protein_id, basedir=C["ELECTROSTATICS_PATH"])
                        f = np.hstack( (phi.reshape(-1,1), nphi.reshape(-1, 1)) )
                        Xe = pnabind.mesh.mapPointFeaturesToMesh(mesh, points, f, map_to="nearest", clip_values=True, laplace_smooth=True, iterations=1)
                        features_e = ["phi", "phi_norm"]
                    except subprocess.CalledProcessError:
                        phi, acc = pnabind.structure.runAPBS(pqr, prefix=protein_id, basedir=C["ELECTROSTATICS_PATH"], **C.get("ELECTROSTATICS_KW", {}))
                        Xe, features_e = pnabind.mesh.mapElectrostaticPotentialToMesh(mesh, phi, acc, efield=True, diff_method='five_point_stencil', sphere_average=True, laplace_smooth=True, scale_to_tabi=True)
                    FEATURE_NAMES += features_e
                    FEATURES.append(Xe)
                os.remove(pqr)
        
        ### LABELS #############################################################################
        if ARGS.labels:
            # Compute labels
            logging.info("Computing a new set of labels for %s" , protein_id)
            protein_int_res = None
            Y = None
            if ARGS.label_method == "structure_structure_distance":
                # get atoms within distance cutoff
                protein_int_res, ligand_int_res = pairsWithinDistance(protein, lig,
                    distance=C.get("ATOM_DISTANCE_CUTOFF", 5.0),
                    skip_hydrogens=True,
                    flatten=True,
                    return_identifier=False,
                    level="R"
                )
                protein_int_res = set(protein_int_res)
                
                # perform laplacian smoothing on residue labels
                if C.get("SMOOTH_LABELS", True):
                    smoothResidues(protein, protein_int_res, mesh=mesh)
                
                # assign labels based on residue list
                Y = pnabind.assignMeshLabelsFromList(protein, mesh, protein_int_res,
                    smooth=False,
                    mask=False,
                    feature_name="binding_site_atom",
                    include_hydrogens=True
                )
                
                # remove binding labels not in L.O.S. of ligand
                if ARGS.check_label_los:
                    Y = checkMeshIntersection(mesh, Y, lig.get_atoms(), distance_cutoff=6.5)
            elif ARGS.label_method == "structure_vertex_distance":
                Y = pnabind.assignMeshLabelsFromStructure(lig, mesh, atom_mapper,
                    smooth=False,
                    distance_cutoff=C.get("MESH_DISTANCE_CUTOFF", 4.5),
                    include_hydrogens=False,
                    mask=False
                )
            
            # Add provided residue id labels if given
            if ARGS.residue_ids:
                binding_residues = set(getResidues(BINDING_RESIDUES_MAP.get(fileName, []), protein))
                if protein_int_res is not None:
                    binding_residues = binding_residues - protein_int_res
                
                # get labels for these given residues
                Yr = pnabind.assignMeshLabelsFromList(protein, mesh, binding_residues,
                    smooth=False,
                    mask=False,
                    feature_name="binding_site_residue",
                    include_hydrogens=True
                )
                
                # combine with previous labels
                if Y is None:
                    Y = Yr
                else:
                    Y = ((Y + Yr) > 0).astype(int)
            
            if C.get("SMOOTH_LABELS", True):
                # apply vertex smoothing
                Y = pnabind.mesh.smoothMeshLabels(mesh.edges, Y, 
                    threshold=C.get("SMOOTHING_THRESHOLD", 150.0),
                    ignore_class=C.get("NO_SMOOTH_CLASSES", [1]),
                    num_classes=2
                )
            
            # Apply masks
            b_mask = None
            if C.get("MASK_LABELS", False):
                b_mask = pnabind.maskClassBoundary(mesh.vertices, Y, mask_cutoff=C.get("MASK_DISTANCE_CUTOFF", 3.0))
            
            c_mask = None
            if MASKED_CHAINS_MAP: 
                mask_residues = []
                # get list of residues in masked chains
                for cid in MASKED_CHAINS_MAP[fileName]:
                    if cid in protein:
                        for residue in protein[cid]:
                            mask_residues.append(residue)
                
                # get labels and convert to mask
                if len(mask_residues) > 0:
                    c_mask = pnabind.assignMeshLabelsFromList(protein, mesh, mask_residues,
                        smooth=False,
                        mask=False,
                        feature_name="mask"
                    )
                    c_mask = c_mask.astype(bool)
                    c_mask = np.logical_not(c_mask)
                else:
                    c_mask = np.ones_like(Y).astype(bool)
        
        ### AUXILIARY DATA #####################################################################
        if ARGS.auxiliary:
            logging.info("Performing auxiliary data computations for %s" , protein_id)
            if not ARGS.labels:
                Y = arrays['Y']
            geo_dist, d_mask = pnabind.mesh.getDistanceFromBoundary(mesh.vertices, mesh.faces, Y)
            arrays['aux_geo_dist'] = geo_dist
            arrays['aux_geo_dist_mask'] = d_mask
        
        ### OUTPUT #############################################################################
        # Write features to disk
        
        # Vertex features
        if ARGS.features:
            FEATURES = [x.reshape(-1, 1) if x.ndim == 1 else x for x in FEATURES]
            arrays['X'] = np.concatenate(FEATURES, axis=1)
            arrays['feature_names'] = np.array(FEATURE_NAMES)
        
        # Mesh labels
        if ARGS.labels:
            arrays[label_key] = Y
            arrays['binding_residues'] = np.array([res.get_full_id() for res in protein_int_res], dtype=object)
            
            if atom_mapper:
                arrays['{}_classes'.format(label_key)] = atom_mapper.classes
            if b_mask is not None:
                arrays['boundary_mask'] = b_mask
            if c_mask is not None:
                arrays['chain_mask'] = c_mask
        
        # Write mesh data to disk
        np.savez_compressed(fname, **arrays)
        logging.info("Saved mesh data to disk: %s", fname)
        if ARGS.output_file is not None:
            PROCESSED.write("{}_data.npz\n".format(protein_id))
        
        # Write mesh adjacency to disk
        if ARGS.adjacency:
            fname = ospj(C['FEATURE_DATA_PATH'], "{}_adj.npz".format(protein_id))
            save_npz(fname, mesh.vertex_adjacency_matrix)
            logging.info("Saved adjacency data to disk: %s", fname)
    
    if ARGS.output_file is not None:
        PROCESSED.close()
    FNULL.close()
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
