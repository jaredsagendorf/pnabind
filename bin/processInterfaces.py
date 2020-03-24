#!/usr/bin/env python

# standard packages
import argparse
import subprocess
import json
import glob
import os
import re
import shutil
import pathlib

# third party packages
import numpy as np
import freesasa
from scipy.sparse import save_npz
from scipy.stats import zscore
from scipy.spatial import ConvexHull, cKDTree
from gridData import Grid
from utils import compileRegexes, Radius, getStructureFromModel, long_to_short
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser, PDBIO, NeighborSearch
from Bio.PDB.Model import Model
from Bio.PDB.DSSP import DSSP
#### Biopython Disordered Atom Fix ####
import Bio.PDB 
copy = Bio.PDB.Atom.copy
def myCopy(self):
    shallow = copy.copy(self)
    for child in self.child_dict.values():
        shallow.disordered_add(child.copy())
    return shallow
Bio.PDB.Atom.DisorderedAtom.copy=myCopy
#### Biopython Disordered Atom Fix ####
import trimesh

# geobind packages
from geobind.preprocessing import runMSMS, runNanoShaper, generateMesh
from Mesh import Mesh, writeOFF


# Command-line arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("structures_file",
                help="a list of interface structures to process")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="file storing configuration options")
arg_parser.add_argument("-s", "--smooth_labels", action='store_true',
                help="perform a smoothing step on labels")
arg_parser.add_argument("-b", "--binary_labels", action='store_true',
                help="use a single class label for DNA binding sites")
arg_parser.add_argument("-r", "--refresh", action='store_true', default=False,
                help="recompute mesh and mesh geometry features")
arg_parser.add_argument("-m", "--mask_labels", action='store_true', default=False,
                help="mask mesh labels which are near bondary of binding site. Masked labels are given a value of -1")
arg_parser.add_argument("-E", "--no_potential", action='store_true',
                help="do not calculate electrostatic potential")
arg_parser.add_argument("-L", "--no_labels", dest="no_labels", action='store_true', default=False,
                help="Do not generate labels for each interface")
arg_parser.add_argument("-F", "--no_features", dest="no_features", action='store_true', default=False,
                help="Do not generate features for each interface")
arg_parser.add_argument("-A", "--no_adjacency", dest="no_adjacency", action='store_true', default=False,
                help="Do not write mesh adjacency matrix to file")
ARGS = arg_parser.parse_args()

# Load the config file
with open(ARGS.config_file) as FH:
    C = json.load(FH)
file_path = pathlib.Path(__file__)
C['ROOT_DIR'] = file_path.parent.parent
DATA_PATH = os.path.join(C['ROOT_DIR'], 'data')

class ResidueMutator(object):
    def __init__(self, candidate_files, components,
        standard_residues=[
            'ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE',
            'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
        ]
    ):
        """ The mutator object takes a non-standard residue or incomplete residue and modifies it
        """
        self.components = components
        self.candidates = {}
        self.standard_residues = standard_residues
        self.imposer = SVDSuperimposer()
        self.parser = PDBParser(PERMISSIVE=1,QUIET=True)
        
        # build up candidate structures
        for fn in candidate_files:
            structure = self.parser.get_structure("", fn)
            resn = structure[0][" "][2].get_resname()
            self.candidates[resn] = []
            for model in structure:
                self.candidates[resn].append(model[" "][2])
    
    def mutate(self, residue, repair=False):
        resn = residue.get_resname()
        if(repair):
            # use residue as its own parent
            parn = resn
        else:
            if(self.standard(resn)):
                # the residue is already a standard residue, do not need to mutate.
                return residue
            parn = self.components[resn]['_chem_comp.mon_nstd_parent_comp_id']
            if(not self.standard(parn)):
                # the parent residue is a nonstandard residue, can't mutate
                return False
        
        if(parn not in self.candidates):
            # parent not in candidate structures
            return False
        
        sc_fixed = set(self.components[resn]['side_chain_atoms']) # side chain atoms of fixed residue
        sc_movin = set(self.components[parn]['side_chain_atoms']) # side chain atoms of standard parent
        atom_names = sc_fixed.intersection(sc_movin)
        
        # get list of side chain atoms present in residue
        atom_list = []
        for atom in atom_names:
            if(atom in residue):
                atom_list.append(atom)
        
        # get side chain atom coordinates
        fixed_coord = np.zeros((len(atom_list), 3))
        for i in range(len(atom_list)):
            fixed_coord[i] = residue[atom_list[i]].get_coord()
        
        # loop over candidates, finding best RMSD
        moved_coord = np.zeros((len(atom_list), 3))
        min_rms = 99999
        rotm = None
        tran = None
        min_candidate = None
        for candidate in self.candidates[parn]:
            for j in range(len(atom_list)):
                moved_coord[j] = candidate[atom_list[j]].get_coord()
            # perfom SVD fitting
            self.imposer.set(fixed_coord, moved_coord)
            self.imposer.run()
            if(self.imposer.get_rms() < min_rms):
                min_rms = self.imposer.get_rms()
                rotm, tran = self.imposer.get_rotran()
                min_candidate = candidate
        
        # copy the candidate to a new object
        candidate = min_candidate.copy()
        candidate.transform(rotm, tran)
        stripHydrogens(candidate)
        
        # replace backbone atoms of candidate
        backbone_atoms = self.components[resn]['main_chain_atoms']
        for atom in backbone_atoms:
            if(atom not in residue):
                continue
            if(atom not in candidate):
                candidate.add(residue[atom].copy())
            candidate[atom].set_coord(residue[atom].get_coord())
        
        return candidate
    
    def standard(self, resname):
        return (resname in self.standard_residues)
    
    def modified(self, resname):
        if(resname in self.standard_residues):
            # it's standard, not modified
            return False
        
        if(resname in self.components and '_chem_comp.mon_nstd_parent_comp_id' in self.components[resname]):
            return (
                (resname not in self.standard_residues)
                and
                (self.components[resname]['_chem_comp.mon_nstd_parent_comp_id'] in self.standard_residues)
            )
        else:
            # has no standard parent field - can't be modified
            return False

class Interpolator(object):
    def __init__(self, fileName):
        self.grid = Grid(fileName) # stores the grid data
            
    def __call__(self, xyz):
        return self.grid.interpolated(xyz[:,0], xyz[:,1], xyz[:,2])

def clipOutliers(data, method="IQR", axis=None):
    if(method == "z-score"):
        Z = np.abs(zscore(data, axis=axis))
        mask = (Z > 3)
    elif(method == "IQR"):
        Q1 = np.quantile(data, 0.25, axis=axis)
        Q3 = np.quantile(data, 0.75, axis=axis)
        IQR = Q3 - Q1 
        mask = np.logical_or((data < (Q1-1.5*IQR)), (data > (Q3+1.5*IQR)))
    mdata = np.ma.array(data, mask=mask)
    lower = mdata.min(axis=axis)
    upper = mdata.max(axis=axis)
    
    return np.clip(data, lower, upper)

def stripHydrogens(structure):
    """Strip all hydrogen atoms from the given model.
    
    Parameters
    ----------
    model: BioPython MODEL object
        The model to be stripped of hydrogens.
    """
    if(structure.get_level() == 'R'):
        rm = []
        for atom in structure:
            if(atom.element == 'H'):
                rm.append(atom.get_id())
        for aid in rm:
            structure.detach_child(aid)
    else:
        for residue in structure.get_residues():
            rm = []
            for atom in residue:
                if(atom.element == 'H'):
                    rm.append(atom.get_id())
            for aid in rm:
                residue.detach_child(aid)

def chainType(chain, regexes):
    dna_count = 0.0
    pro_count = 0.0
    for residue in chain:
        resname = residue.get_resname().strip()
        if(regexes['PROTEIN']['STANDARD_RESIDUES'].search(resname)):
            pro_count += 1.0
        if(regexes['DNA']['STANDARD_NUCLEOTIDES'].search(resname)):
            dna_count += 1.0
    
    if(dna_count > pro_count):
        return 'dna'
    else:
        return 'pro'
    
def getEntities(structure, regexes, mi=0):
    pro = Model(mi)
    dna = Model(mi)
    
    for chain in structure[0]:
        ctype = chainType(chain, regexes)
        if(ctype == "dna"):
            dna.add(chain)
        if(ctype == "pro"):
            pro.add(chain)
    
    return pro, dna

def getAtomParameters(prefix, model, hydrogens=True, keepPQR=False):
    """ Assign atomic parameters to every atom in a protein chain. Values are stored in atom.xtra 
    dictionary. The following keys are used
    ------------------------------------------------------------------------------------------------
    radius - van der Waals radius
    charge - atomic effective charge
    """
    io = PDBIO()
    
    # Strip any existing hydrogens - add new ones with PDB2PQR
    stripHydrogens(model)
    
    # Write chain to temp file
    pdbFile = "{}_temp.pdb".format(prefix)
    pqrFile = "{}.pqr".format(prefix)
    io.set_structure(model)
    io.save(pdbFile)
    
    # Run PDB2PQR
    FNULL = open(os.devnull, 'w')
    subprocess.call([
            'pdb2pqr',
            '--ff=amber',
            '--chain',
            pdbFile,
            pqrFile
        ],
        stdout=FNULL,
        stderr=FNULL
    )
    FNULL.close()
    
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = parser.get_structure("repaired", pqrFile)
    model = structure[0]
    
    # Get radius and charge from PQR file
    for line in open(pqrFile):
        if(line[0:4] != "ATOM"):
            continue
        cid = line[21]
        num = int(line[22:26].strip())
        ins = line[26]
        rid = (" ", num, ins)
        crg = float(line[55:62].strip())
        vdw = float(line[63:69].strip())
        atm = line[12:16].strip()
        if(vdw == 0.0):
            vdw = 0.6 # 0 radius atoms causes issues - set to a minimum of 0.6
        if(rid in model[cid]):
            if(atm in model[cid][rid]):
                model[cid][rid][atm].xtra["charge"] = crg 
                model[cid][rid][atm].xtra["radius"] = vdw
    
    if(not hydrogens):
        # remove hydrogens
        stripHydrogens(model)
    
    # clean up
    os.remove(pdbFile)
    if(not keepPQR):
        os.remove(pqrFile)
    
    return model

def cleanProtein(model, mutator, REGEXES, hydrogens=True):
    """ Perform any operations needed to modify the structure or sequence of a protein
    chain.
    """
    # remove hydrogens if requested
    if(not hydrogens):
        stripHydrogens(model)
    
    # remove non-standard residues
    for chain in model:
        replace = []
        remove = []
        for residue in chain:
            resn = residue.get_resname().strip()
            if(mutator.standard(resn)):
                continue
            elif(resn == 'HOH' or resn == 'WAT'):
                remove.append(residue.get_id())
            elif(REGEXES["SOLVENT_COMPONENTS"].search(resn)):
                continue
            elif(mutator.modified(resn)):
                replace.append(residue.get_id())
            else:
                remove.append(residue.get_id())
        
        for rid in remove:
            print("removed unrecognized residue: %s" % (chain[rid].get_resname()))
            chain.detach_child(rid)
        
        for rid in replace:
            print("replacing modified residue: %s" % (chain[rid].get_resname()))
            replacement = mutator.mutate(chain[rid])
            chain.detach_child(rid)
            if(replacement):
                replacement.id = rid
                chain.add(replacement)
    
    return model

def computeCV(model, residues, ns, radius, key="cv", hydrogens=False, bonds=None):
    """ get CV values for every residue in residues"""
    
    if(not hydrogens and bonds is None):
        raise ValueError("must provide bond data if hydrogens=False!")
    
    for resID in residues:
        cid, num, ins = resID.split('.')
        residue = model[cid][(' ', int(num), ins)]
        for atom in residue:
            if(not hydrogens and atom.element == 'H'):
                continue
            vector = np.zeros(3)
            neighbors = ns.search(atom.get_coord(), radius, level='A')
            for n in neighbors:
                if(n == atom):
                    continue
                if(not hydrogens and n.element == 'H'):
                    continue
                vector += (atom.get_coord() - n.get_coord())/np.linalg.norm(atom.get_coord() - n.get_coord())
            atom.xtra[key] = (1 - np.linalg.norm(vector)/(len(neighbors)-1))
    
    # use parent atom as hydrogen CV value if we exluded them
    if(not hydrogens):
        for resID in residues:
            cid, num, ins = resID.split('.')
            residue = model[cid][(' ', int(num), ins)]
            resn = residue.get_resname().strip()
            for atom in residue:
                if(atom.element != 'H'):
                    continue
                aname = atom.get_name().strip()
                parent_atom = bonds[resn][aname]['bonded_atoms'][0]
                atom.xtra[key] = residue[parent_atom].xtra[key]

def getSAP(model, ns, regexes, residue_hydrophobicity, standard_area,
        distance=5.0, area_key='sesa', hydrogens=False, bonds=None
    ):
    # Reads in the file pdbid-protein.pdb and computes the SAP score for
    # all standard residues on the protein surface. Non-standard should 
    # be assigned a score of none. Non-standard residue atoms are ignored
    # and are not considered in SAP calculations.
    # Arguments:
    # pdbid:       structure name/identifier
    # distance:    distance cut-off for neighbor search
    #-------------------------------------------------------------------
    
    if(not hydrogens and bonds is None):
        raise ValueError("must provide bond data if hydrogens=False!")
    
    # compute the atom SAP scores
    for chain in model:
        for residue in chain:
            for a in residue:
                if(not hydrogens and a.element == 'H'):
                    continue
                if(a.xtra[area_key] <= 0.0):
                    a.xtra["sap"] = 0.0
                    continue
                center = a.get_coord()
                sap = 0.0
                neighbors = ns.search(center, distance, level='A')
                for n in neighbors:
                    if(not hydrogens and n.element == 'H'):
                        continue
                    if(n.xtra[area_key] <= 0.0):
                        continue
                    nname = n.get_name().strip()
                    nres = n.get_parent()
                    nresn = nres.get_resname().strip()
                    # select side-chain atoms only
                    if(regexes['PROTEIN']['SIDE_CHAIN'].search(nname)):
                        if(standard_area[nresn][nname] == 0.0):
                            sap += residue_hydrophobicity[nresn] # assume a RASA of 1.0
                        else:
                            sap += residue_hydrophobicity[nresn]*min(1.5, n.xtra[area_key]/standard_area[nresn][nname])
                a.xtra["sap"] = sap
    
    # use parent atom as hydrogen SAP value if we exluded them
    if(not hydrogens):
        for chain in model:
            for residue in chain:
                resn = residue.get_resname().strip()
                for a in residue:
                    if(a.element != 'H'):
                        continue
                    aname = a.get_name().strip()
                    parent_atom = bonds[resn][aname]['bonded_atoms'][0]
                    a.xtra["sap"] = residue[parent_atom].xtra["sap"]

def getDSSP(model, fileName, prefix='ss_'):
    # run DSSP using the DSSP class from BioPython
    dssp = DSSP(model, fileName)
    
    # store secondary structure in each atom
    for chain in model:
        cid = chain.get_id()
        for residue in chain:
            rid = residue.get_id()
            dkey = (cid, rid)
            if(dkey in dssp):
                ss = C["DSSP_MAP"][dssp[dkey][2]]
            else:
                ss = 'L'
            for atom in residue:
                atom.xtra[prefix+'H'] = 0.0
                atom.xtra[prefix+'S'] = 0.0
                atom.xtra[prefix+'L'] = 0.0
                atom.xtra[prefix + ss] = 1.0

def getHBondPotential(X, vi, atom, mesh, hb_info, cutoff=2.6, da_cutoff=3.9):
    aname = atom.name
    parent = atom.get_parent()
    parent_name = parent.get_resname()
    
    if(parent_name in hb_info):
        # Donor Atoms
        if(aname in hb_info[parent_name]['donors']):
            for h in hb_info[parent_name]['donors'][aname]['hydrogen_atoms']:
                # loop over hydrogen atoms
                if(h in parent):
                    hatom = parent[h]
                    hv = mesh.V[vi] - hatom.get_coord() # H->V vectors
                    #ha = atom.get_coord() - hatom.get_coord() # H->D vector
                    #hv_n = np.linalg.norm(hv, axis=1) # norm of the H->V vectors
                    #ha_n = np.linalg.norm(ha) # norm of the H-> D vector
                    
                    # compute distances and angles
                    h_dist = np.linalg.norm(hv, axis=1) # distance from V to H
                    score = 2*np.arctan(-(h_dist-cutoff))/np.pi
                    #h_cos_angle = (hv*ha).sum(axis=1)/(ha_n*hv_n) # cosine of angle between VH and DH
                    
                    #mask = (h_dist <= hd_cutoff)
                    #X[vi, 0] += np.clip(mask*h_cos_angle*((hd_cutoff/h_dist)**2 - 1), 0, 1)
                    X[vi, 0] += np.clip(score, 0.0, 1.0)
        
        # Acceptor Atoms
        if(aname in hb_info[parent_name]['acceptors']):
            if(parent_name == 'HIS'):
                # check protonation state
                if(aname == 'ND1' and 'HD1' in parent):
                    return 
                if(aname == 'NE2' and 'HE2' in parent):
                    return
            
            av = mesh.V[vi] - atom.get_coord() # A->V vectors
            a_dist = np.linalg.norm(av, axis=1) # distance from V to A
            score = 2*np.arctan(-(a_dist-cutoff))/np.pi
            X[vi, 1] += np.clip(score, 0.0, 1.0)

#def getHBondInfo(model, hbond_data, hydrogens=False):
    #for chain in model:
        #for residue in chain:
            #rname = residue.get_resname()
            #for atom in residue:
                #aname = atom.get_id()
                #if(hydrogens):
                    #if(aname not in hbond_data[rname]):
                        #print(rname, aname, atom.get_parent().get_id())
                    #atom.xtra['hb_D'] = 0.0
                    #atom.xtra['hb_A'] = 0.0
                    #atom.xtra['hb_H'] = 0.0
                    #atom.xtra['hb_X'] = 0.0
                    #atom.xtra['hb_'+ hbond_data[rname][aname]["hydrogen_bond_status"]] = 1.0
                #else:
                    #atom.xtra['hb_D'] = 0.0
                    #atom.xtra['hb_A'] = 0.0
                    #atom.xtra['hb_X'] = 0.0
                    #atom.xtra['hb_'+ hbond_data[rname][aname]["hydrogen_bond_status"]] = 1.0
    
def getAtomSESA(prefix, chain_atoms, clean=True, hydrogens=False):
    # run MSMS
    af = runMSMS(chain_atoms, prefix, '.', area_only=True, hydrogens=hydrogens)
    
    # read in area file
    SE = open(af)
    SE.readline()
    count = 0
    for i in range(len(chain_atoms)):
        if((not hydrogens) and chain_atoms[i].element == 'H'):
            continue
        sesa = float(SE.readline().strip().split()[1])
        chain_atoms[i].xtra['sesa'] = sesa
        count += 1
    
    # clean up
    if(clean):
        os.remove(af)

def getAtomSASA(model, classifier=freesasa.Classifier(), probe_radius=1.4):
    structure = getStructureFromModel(model, classifier=classifier)
    SASA = freesasa.calc(structure, freesasa.Parameters({"probe-radius": probe_radius}))
    
    # get atom SASA
    N = structure.nAtoms()
    for i in range(N):
        sasa = SASA.atomArea(i)
        resi = structure.residueNumber(i).strip()
        if(resi[-1].isdigit()):
            ins = " "
        else:
            ins = resi[-1]
            resi = resi[:-1]
        aname = structure.atomName(i).strip()
        chain[(' ', int(resi), ins)][aname].xtra["sasa"] = sasa

def getConvexHull(points, prefix, directory='.'):
    hull = ConvexHull(points)
    
    v = points[hull.vertices]
    f = hull.simplices
    
    # renumber face indices
    vm = np.zeros(points.shape[0])
    vm[hull.vertices] = np.arange(v.shape[0])
    f[:,0] = vm[f[:,0]]
    f[:,1] = vm[f[:,1]]
    f[:,2] = vm[f[:,2]]
    
    fileName = os.path.join(directory, prefix)
    writeOFF(fileName, v, f)
    
    return fileName+'.off'

#def getAtomExposure(chain_atoms, standard_area, area_key='sesa', hydrogens=True):
    #for atom in chain_atoms:
        #aname = atom.name
        #aresn = atom.get_parent().get_resname()
        #for res in standard_area:
            #atom.xtra['{}_RASA'.format(res)] = 0.0
        #if(not hydrogens and atom.element == "H"):
            #continue
        #if(aname not in standard_area[aresn]):
            #print(aresn, aname, atom.get_parent().get_id())
        #if(standard_area[aresn][aname] == 0.0):
            #atom.xtra['{}_RASA'.format(aresn)] = 1.0
        #else:
            #atom.xtra['{}_RASA'.format(aresn)] = min(1.5, atom.xtra[area_key]/standard_area[aresn][aname])

def getSurfaceResidues(chain_atoms, area_key='sesa', threshold=0.0, hydrogens=False):    
    # get residue area values
    residueArea = {} # dict to map rid to an area value
    for atom in chain_atoms:
        if((not hydrogens) and atom.element == 'H'):
            continue
        aid = atom.get_full_id()
        rid = "{}.{}.{}".format(aid[2], aid[3][1], aid[3][2])
        if(rid not in residueArea):
            residueArea[rid] = 0.0
        residueArea[rid] += atom.xtra[area_key]
    
    # determine surface residues
    surface_residues = []
    for rid in residueArea:
        if(residueArea[rid] > threshold):
            surface_residues.append(rid)
    
    return surface_residues

def getChargeDensity(atoms, vertices):
    q = []
    R = []
    x = []
    for a in atoms:
        q.append(a.xtra["charge"])
        R.append(a.xtra["radius"])
        x.append(a.get_coord())
    #print(q)
    #print(R)
    q = np.array(q) # array of atomic charge of size A
    R = np.array(R) # array of atomic radius of size A
    x = np.array(x) # array of atomic position of size A
    
    s = q/R # array of charge/Radius of size A
    #print(s)
    #exit(0)
    d = np.linalg.norm(vertices[:, np.newaxis] - x , axis=2) # V x A array of vertex-atom distances
    
    rho = (s*np.exp(-0.5*((d-R)/R)**2)).sum(axis=1) # array of charge-density of size V
    
    return rho

def runAPBS(model, prefix="tmp", basedir='.', quiet=True, pqr=None, clean=True):
    """ run APBS and return potential """
    if(pqr is None):
        tmp = os.path.join(basedir, "{}.pdb".format(prefix))
        pqr = os.path.join(basedir, "{}.pqr".format(prefix))
        FNULL = open(os.devnull, 'w')
    
        # write the chain to file
        io.set_structure(model)
        io.save(tmp)
    
        # run PDB2PQR
        subprocess.call([
            'pdb2pqr',
            '--ff=amber',
            '--chain',
            tmp,
            pqr
            ],
            stdout=FNULL,
            stderr=subprocess.STDOUT
        )
        FNULL.close()
    
    # APBS will have issues reading PQR file if coordinate fields touch
    padCoordinates(pqr)
    
    # run psize to get grid length parameters
    stdout = subprocess.getoutput("psize --space 0.3 '{}'".format(pqr))
    cglenMatch = re.search('Coarse grid dims = (\d*\.?\d+) x (\d*\.?\d+) x (\d*\.?\d+) A', stdout, re.MULTILINE)
    cgx = cglenMatch.group(1)
    cgy = cglenMatch.group(2)
    cgz = cglenMatch.group(3)
    fglenMatch = re.search('Fine grid dims = (\d*\.?\d+) x (\d*\.?\d+) x (\d*\.?\d+) A', stdout, re.MULTILINE)
    fgx = fglenMatch.group(1)
    fgy = fglenMatch.group(2)
    fgz = fglenMatch.group(3)
    dimeMatch = re.search('Num. fine grid pts. = (\d+) x (\d+) x (\d+)', stdout, re.MULTILINE)
    dx = dimeMatch.group(1)
    dy = dimeMatch.group(2)
    dz = dimeMatch.group(3)
    
    # run APBS
    pot = os.path.join(basedir, prefix+"_potential")
    acc = os.path.join(basedir, prefix+"_access")
    input_file = """READ
    mol pqr {}
END

ELEC
    mg-auto
    mol 1
    
    dime {}
    cglen {}
    fglen {}
    cgcent mol 1
    fgcent mol 1
    
    lpbe
    bcfl sdh
    pdie 2.0
    sdie 78.0
    srfm smol
    chgm spl2
    sdens 10.00
    srad 1.40
    swin 0.30
    temp 310.0
    
    ion charge +1 conc 0.15 radius 2.0
    ion charge -1 conc 0.15 radius 1.8
    
    write pot dx {}
    write smol dx {}
END""".format(
        pqr,
        " ".join([dx, dy, dx]),
        " ".join([cgx, cgy, cgz]),
        " ".join([fgx, fgy, fgz]),
        pot,
        acc
    )
    inFile = os.path.join(basedir, "{}.in".format(prefix))
    FH = open(inFile, "w")
    FH.write(input_file)
    FH.close()
    
    if(quiet):
        FNULL = open(os.devnull, 'w')
        subprocess.call(["apbs", inFile], 
            stdout=FNULL,
            stderr=subprocess.STDOUT
        )
        FNULL.close()
    else:
        print("Running APBS")
        subprocess.call(["apbs", inFile])
    
    # cleanup
    if(clean):
        if(os.path.exists(os.path.join(basedir, "{}.pdb".format(prefix)))):
            os.remove(os.path.join(basedir, "{}.pdb".format(prefix)))
        os.remove(inFile)
        if(os.access('io.mc', os.R_OK)):
            os.remove('io.mc')
    
    return Interpolator("{}.dx".format(pot)), Interpolator("{}.dx".format(acc)),

def getAchtleyFactors(atom_list):
    """Citation: www.pnas.org/cgi/doi/10.1073/pnas.0408677102"""
    achtley_factors = {
        'A': [-0.591, -1.302, -0.733, 1.570, -0.146],
        'C': [-1.343, 0.465, -0.862, -1.020, -0.255],
        'D': [1.050, 0.302, -3.656, -0.259, -3.242],
        'E': [1.357, -1.453, 1.477, 0.113, -0.837],
        'F': [-1.006, -0.590, 1.891, -0.397, 0.412],
        'G': [-0.384, 1.652, 1.330, 1.045, 2.064],
        'H': [0.336, -0.417, -1.673, -1.474, -0.078],
        'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
        'K': [1.831, -0.561, 0.533, -0.277, 1.648],
        'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
        'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
        'N': [0.945, 0.828, 1.299, -0.169, 0.933],
        'P': [0.189, 2.081, -1.628, 0.421, -1.392],
        'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
        'R': [1.538, -0.055, 1.502, 0.440, 2.897],
        'S': [-0.228, 1.399, -4.760, 0.670, -2.647],
        'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
        'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
        'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
        'Y': [0.260, 0.830, 3.097, -0.838, 1.512]
    }
    
    for atom in atom_list:
        resn = atom.get_parent().get_resname().strip()
        resn_short = long_to_short[resn]
        
        for i in range(5):
            atom.xtra['achtley_factor_{}'.format(i+1)] = achtley_factors[resn_short][i]

def getPocketData(atom_list, mesh, X, nn_cutoff=1.5, radius_big=3.0, clean=True, offset=0):
    # run NanoShaper
    runNanoShaper(atom_list, "pockets", ".", pockets_only=True, kwargs=dict(radius_big=radius_big))
    
    # gather pocket meshes
    pockets = glob.glob("cav_tri*.off")
    if(len(pockets) > 0):
        for p in pockets:
            pocket = Mesh(p)
            for i in range(pocket.nV):
                # find adjacent vertices in mesh
                v, d = mesh.findVerticesInBall(pocket.V[i], nn_cutoff)
                if(len(v) > 0):
                    X[v,0+offset] = pocket.volume
                    X[v,1+offset] = pocket.area
                    X[v,2+offset] = pocket.volume/pocket.area
                    X[v,3+offset] = pocket.bbox.aspect_ratio
    
    if(clean):
        # remove all the stuff NanoShaper generated
        files = ["numcav.txt", "cavities.txt","cavitiesSize.txt", "cavAtomsSerials.txt" "triangulatedSurf.off"]
        files += glob.glob("all_cav*.txt")
        files += glob.glob("cav*.txt")
        files += glob.glob("cav_tri*.off")
        for f in files:
            if(os.path.exists(f)):
                os.remove(f)

def padCoordinates(pqrFile):
    padded = open("tmp.pqr", "w")
    with open(pqrFile) as FH:
        for line in FH:
            s = line[:30]
            e = line[54:]
            x = line[30:38].strip()
            y = line[38:46].strip()
            z = line[46:54].strip()
            padded.write("{}{:>9s}{:>9s}{:>9s}{}".format(s, x, y, z, e))
    padded.close()
    shutil.move("tmp.pqr", pqrFile)

def oneHotEncode(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def writePDBFile(model, prefix):
    io = PDBIO()
    
    # Write chain to file
    pdbFile = "{}.pdb".format(prefix)
    io.set_structure(model)
    io.save(pdbFile)
    
    return pdbFile

def generateSpherePoints(n, r=1):
    """Implemented from note by Markus Deserno: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf"""
    alpha = 4.0*np.pi*r*r/n
    d = np.sqrt(alpha)
    
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    xp = []
    yp = []
    zp = []
    for m in range (0,m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range (0,m_phi):
            phi = 2*np.pi*n/m_phi
            xp.append(r*np.sin(nu)*np.cos(phi))
            yp.append(r*np.sin(nu)*np.sin(phi))
            zp.append(r*np.cos(nu))
    
    return np.stack([xp, yp ,zp], axis=-1)

#def signedVolume2(a, b, c, d):
    #return np.sum((a-d)*np.cross(b-d, c-d))
    
#def segmentsIntersectTriangles2(s, t):
    #ind = []
    #normals = np.cross(t[2]-t[0], t[2]-t[1])
    #normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    
    #for i in range(len(s[0])):
        #count = 0 # number of triangle crossings
        #for j in range(len(t[0])):
            #s1 = np.sign(np.sum(normals[j]*(s[0][i] - t[2][j])))
            #s2 = np.sign(np.sum(normals[j]*(s[1][i] - t[2][j])))
            #if(s1 != s2 and s1 != 0 and s2 != 0):
                ## crosses the plane, compute volumes
                #v1 = np.sign(signedVolume2(t[0][j], t[1][j], s[0][i], s[1][i]))
                #v2 = np.sign(signedVolume2(t[1][j], t[2][j], s[0][i], s[1][i]))
                #v3 = np.sign(signedVolume2(t[2][j], t[0][j], s[0][i], s[1][i]))
                
                #if(v1 == v2 and v2 == v3):
                    ## crosses this triangle
                    #count +=1
        #if(count == 0):
            #ind.append(i)
    
    #return ind

def signedVolume(a, b, c, d):
    """Computes the signed volume of a series of tetrahedrons defined by the vertices in 
    a, b c and d. The ouput is an SxT array which gives the signed volume of the tetrahedron defined
    by the line segment 's' and two vertices of the triangle 't'."""
    
    return np.sum((a-d)*np.cross(b-d, c-d), axis=2)

def segmentsIntersectTriangles(s, t):
    """For each line segment in 's', this function computes whether it intersects any of the triangles
    given in 't'."""
    # compute the normals to each triangle
    normals = np.cross(t[2]-t[0], t[2]-t[1])
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    
    # get sign of each segment endpoint, if the sign changes then we know this segment crosses the
    # plane which contains a triangle. If the value is zero the endpoint of the segment lies on the 
    # plane.
    # s[i][:, np.newaxis] - t[j] -> S x T x 3 array
    sign1 = np.sign(np.sum(normals*(s[0][:, np.newaxis] - t[2]), axis=2)) # S x T
    sign2 = np.sign(np.sum(normals*(s[1][:, np.newaxis] - t[2]), axis=2)) # S x T
        
    # determine segments which cross the plane of a triangle. 1 if the sign of the end points of s is 
    # different AND one of end points of s is not a vertex of t
    cross = (sign1 != sign2)*(sign1 != 0)*(sign2 != 0) # S x T 
    
    # get signed volumes
    v1 = np.sign(signedVolume(t[0], t[1], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v2 = np.sign(signedVolume(t[1], t[2], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v3 = np.sign(signedVolume(t[2], t[0], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    
    same_volume = np.logical_and((v1 == v2), (v2 == v3)) # 1 if s and t have same sign in v1, v2 and v3
    
    return np.nonzero(np.logical_not(np.sum(cross*same_volume, axis=1)))[0]

def smoothLabels(mesh, key, class_label=1, threshold=16.0):
    #### generate face labels taking majority vote of vertex labels
    y_face = ((mesh.vertex_attributes[key][mesh.faces] == class_label).sum(axis=1) >= 2).astype(np.int32)
    Nf = y_face.shape[0]
    node_mask = (y_face == 1)
    Nc = node_mask.sum()
    
    #### create a graph with <class> faces as vertices and connect two vertices if the faces share an edge. 
    E_face = mesh.face_adjacency
    edge_mask = node_mask[E_face[:,0]]*node_mask[E_face[:,1]] # edges where both nodes are a <class> face
    
    # create an index from 0..Nf-1 to 0..Nc-1 when we apply the node mask or edge mask
    map_c = np.empty(Nf, dtype=np.int32)
    map_c[node_mask] = np.arange(Nc)
    
    # map the <class> edges to be within range of 0..Nc-1 and make undirected
    c_edges = map_c[E_face[edge_mask]]
    e1, e2 = c_edges[:, 0], c_edges[:, 1]
    e1, e2 = np.hstack((e1, e2)), np.hstack((e2, e1))
    c_edges = np.stack((e1, e2), axis=1)
    
    # get <class> nodes from 0..Nc-1
    c_nodes = map_c[node_mask]
    
    #### find all connected components in the <class> faces graph
    components = trimesh.graph.connected_components(c_edges, min_len=0, nodes=c_nodes, engine='scipy')
    
    map_a = np.argwhere(node_mask).flatten() # index to the the original face indices
    component_sizes = np.array([mesh.area_faces[map_a[c]].sum() for c in components])
    
    #### flip labels where component_size < threshold (total triangle area)
    components_to_flip = np.argwhere(component_sizes < threshold).flatten()
    vertices = []
    for ci in components_to_flip:
        face_idx = map_a[components[ci]]
        vertices.append(mesh.faces[face_idx].flatten())
    
    if(len(vertices) > 0):
        vertices = np.hstack(vertices)
        mesh.vertex_attributes[key][vertices] = 1 - class_label

def main():
    ### Load various data files ####################################################################
    tripeptides = glob.glob(os.path.join(DATA_PATH, "tripeptides/*_md.pdb"))
    with open(os.path.join(DATA_PATH, 'components.json')) as FILE:
        components = json.load(FILE)
    pdb_parser = PDBParser(PERMISSIVE=1,QUIET=True)
    mutator = ResidueMutator(tripeptides, components)
    
    # Regular expressions
    with open(os.path.join(DATA_PATH, 'regexes.json')) as FILE:
        regexes = json.load(FILE)
        compileRegexes(regexes)
    
    # Residue hydrophobicity data
    with open(os.path.join(DATA_PATH,'residue-hydrophobicity.json')) as FILE:
        residue_hydrophobicity = json.load(FILE)
    
    # Get standard area file
    if(C["AREA_MEASURE"] == "sasa"):
        classifier = Radius()
        classifier.initialize(DATA_PATH, components)
        saf = 'standard-sasa.json'
    else:
        saf = 'standard-sesa.json'
    #if(C["HYDROGENS"]):
        #if(C["AREA_MEASURE"] == "sesa"):
            #saf = 'standard-sesa-H.json'
        #else:
            #saf = 'standard-sasa-H.json'
    #else:
        #if(C["AREA_MEASURE"] == "sesa"):
            #saf = 'standard-sesa.json'
        #else:
            #saf = 'standard-sasa.json'
    with open(os.path.join(DATA_PATH, saf)) as FILE:
        standard_area = json.load(FILE)
    
    # Get hydrogen bond donor/acceptor data
    with open(os.path.join(DATA_PATH,'hbond-data.json')) as FILE:
        hbond_data = json.load(FILE)
    
    # Get covalent bond data
    with open(os.path.join(DATA_PATH,'bond-data.json')) as FILE:
        bond_data = json.load(FILE)
    
    # Load the interface file which describes a list of DNA-protein interfaces to process
    listFile = ARGS.structures_file
        
    for fileName in open(listFile):
        fileName = fileName.strip()
        if(fileName[0] == '#'):
            # skip commented lines
            continue
        structure = pdb_parser.get_structure('structure', os.path.join(C["PDB_FILES_PATH"], fileName))
        protein_id = '.'.join(fileName.split('.')[0:-1]) + '_protein'
        protein, dna = getEntities(structure, regexes)
            
        ### MESH GENERATION ####################################################################
        # Clean the protein entity
        protein = cleanProtein(protein, mutator, regexes, hydrogens=C['HYDROGENS'])
        
        # Get radii/charge parameters for chain
        protein = getAtomParameters(protein_id, protein, keepPQR=True, hydrogens=C['HYDROGENS'])
        
        # Write a PDB file matching chain
        pdb = writePDBFile(protein, protein_id)
        
        # Generate a mesh
        mesh_prefix = "{}_mesh".format(protein_id)
        if(ARGS.refresh):
            mesh = generateMesh(protein, 
                prefix=mesh_prefix,
                basedir=C["MESH_FILES_PATH"],
                kwargs=dict(op_mode='normal', surface_type='skin', skin_parameter=0.45, grid_scale=C.get("GRID_SCALE", 2.0))
            )
            print("Computed new mesh for: {}".format(protein_id))
            meshFile = mesh.save(C['MESH_FILES_PATH'], overwrite=True)
        else:
            meshFile = os.path.join(C['MESH_FILES_PATH'], "{}_mesh.off".format(protein_id))
            if(os.path.exists(meshFile)):
                mesh = Mesh(meshFile, name=mesh_prefix)
                print("Loaded existing mesh for: {}".format(protein_id))
            else:
                # couldn't find mesh files, compute from scratch
                mesh = generateMesh(protein, 
                    prefix=mesh_prefix,
                    basedir=C["MESH_FILES_PATH"],
                    kwargs=dict(op_mode='normal', surface_type='skin', skin_parameter=0.45)
                )
                print("Computed new mesh for: {}".format(protein_id))
                meshFile = mesh.save(C['MESH_FILES_PATH'])
        
        ### FEATURES ###########################################################################
        if(not ARGS.no_features):
            # Generate a convex hull
            hullFile = getConvexHull(mesh.V, "{}_hull".format(protein_id), C["MESH_FILES_PATH"])
            
            # Compute mesh geometry features. These features are computed only from the mesh and are
            # independent of the protein structure.
            if(ARGS.refresh):
                # recompute mesh features
                args = ['computeMeshFeatures', protein_id, meshFile, hullFile]
                subprocess.call(args)
            else:
                if(not os.path.exists("{}_mesh_features.dat".format(protein_id))):
                    # couldn't find mesh features file, compute from scratch
                    args = ['computeMeshFeatures', protein_id, meshFile, hullFile]
                    subprocess.call(args)
            Xm = np.loadtxt("{}_mesh_features.dat".format(protein_id), dtype=np.float32)
            mesh.Nv = "{}_normals.dat".format(protein_id)
            
            # Check that the size of the feature data matches the mesh size
            if(Xm.shape[0] != mesh.nV):
                raise ValueError("The sizes of the mesh feature data and the mesh do not match! ({}, {})".format(Xm.shape[0], mesh.nV))
            features_m = [_.strip() for _ in open("feature_names.dat").readlines()]
            
            # Compute atom-level features and store them in atom.xtra of chain. Atom-level features 
            # are computed based only on the protein structure and are independent of the mesh.
            pro_atoms = [atom for atom in protein.get_atoms()]
            atomKDTree = NeighborSearch(pro_atoms)
            if(C["AREA_MEASURE"] == "sesa"):
                getAtomSESA(protein_id, pro_atoms)
            else:
                getAtomSASA(protein, classifier=classifier)
            surface_residues = getSurfaceResidues(pro_atoms, hydrogens=False)
            
            # Atom-level features names
            features_a = ["cv_fine", "cv_medium", "cv_coarse", "sap"]
            for i in range(5):
                features_a.append("achtley_factor_{}".format(i+1))
            #for res in standard_area:
                #features_a.append("{}_RASA".format(res)) # atom relative area
            for ss in ['H', 'S', 'L']:
                features_a.append("ss_{}".format(ss)) # residue secondary structure
            
            #getAtomExposure(pro_atoms, standard_area, hydrogens=C["HYDROGENS"])
            #getHBondInfo(protein, hbond_data, hydrogens=C["HYDROGENS"])
            getSAP(protein, atomKDTree, regexes, residue_hydrophobicity, standard_area, distance=5.0, hydrogens=False, bonds=bond_data)
            computeCV(protein, surface_residues, atomKDTree, 10.00, key=features_a[0], hydrogens=False, bonds=bond_data)
            computeCV(protein, surface_residues, atomKDTree, 25.00, key=features_a[1], hydrogens=False, bonds=bond_data)
            computeCV(protein, surface_residues, atomKDTree, 100.0, key=features_a[2], hydrogens=False, bonds=bond_data)
            getDSSP(protein, pdb)
            getAchtleyFactors(pro_atoms)
            
            # Map atom-level features to the mesh, weighted by inverse distance from the atom to 
            # nearby mesh vertices
            
            # atom features and weights
            Xa = np.zeros((mesh.nV, len(features_a)))
            wa = np.zeros(mesh.nV) # distance-based weight sums
            
            # features which depend on both the atomic coordinates and the 
            # mesh coordinates.
            features_h = ['hb_donor', 'hb_acceptor']
            Xh = np.zeros((mesh.nV, len(features_h)))
            
            # loop over surface residue atoms
            for atom in pro_atoms:
                if(not C["HYDROGENS"] and atom.element == 'H'):
                    # Don't map hydrogen properties to the mesh
                    continue
                aid = atom.get_full_id()
                rid = "{}.{}.{}".format(aid[2], aid[3][1], aid[3][2])
                
                if(rid in surface_residues):
                    v, d = mesh.findVerticesInBall(atom.coord, C["ATOM_MESH_DISTANCE_CUTOFF"])
                    if(len(v) == 0):
                        # increase the search distance a bit
                        v, d = mesh.findVerticesInBall(atom.coord, C["ATOM_MESH_DISTANCE_CUTOFF"]+1.0)
                    if(len(v) > 0):
                        w = np.clip(1/(d+1e-5), 0.0, 2.0)
                        data = list(map(lambda f: atom.xtra[f], features_a))
                        Xa[v] += np.outer(w, data)
                        wa[v] += w
                    
                    # hydrogen bond potential
                    getHBondPotential(Xh, v, atom, mesh, hbond_data)
            # set zero weights to 1
            wi = (wa == 0)
            wa[wi] = 1.0
            # scale by weights
            Xa /= wa.reshape(-1, 1)
            Xh = np.clip(Xh, 0.0, 1.0)
            
            # Compute pocket features
            features_p = ["pocket_volume", "pocket_area", "pocket_V/A", "pocket_aspect_ratio"]
            Xp = np.zeros((mesh.nV, len(features_p)))
            getPocketData(pro_atoms, mesh, Xp, radius_big=3.0, offset=0)
            
            # Compute Electrostatic features
            if(not ARGS.no_potential):
                features_e = ['averaged_potential']
                Xe = np.zeros((mesh.nV, len(features_e)))
                potfile = os.path.join(C["ELECTROSTATICS_PATH"], protein_id+"_potential.dx")
                accessfile = os.path.join(C["ELECTROSTATICS_PATH"], protein_id+"_access.dx")
                if((not ARGS.refresh) and os.path.exists(potfile) and os.path.exists(accessfile)):
                    print("Loaded potential {} from file.".format(potfile))
                    print("Loaded accessibility {} from file.".format(accessfile))
                    phi = Interpolator(potfile)
                    acc = Interpolator(accessfile)
                else:
                    phi, acc = runAPBS(protein, protein_id, pqr="{}.pqr".format(protein_id), basedir=C["ELECTROSTATICS_PATH"])
                
                # compute point cloud
                kernel = generateSpherePoints(50, r=1.0) # unit sphere at the origin
                
                # sample over kernel
                points = (mesh.V[:, np.newaxis] + kernel).reshape(-1, 3) # V*K x 3 array of points
                phi_s = phi(points).reshape(mesh.nV, -1) # V x K potential samples
                acc_s = acc(points).reshape(mesh.nV, -1) # V x K accessibility samples
                phi_s = phi_s*acc_s # masking inaccessible potential values
                phi_s = phi_s.sum(axis=1)/acc_s.sum(axis=1) # V array of averaged potential
                
                Xe[:, features_e.index('averaged_potential')] += clipOutliers(phi_s)
            
            # Clean-up
            os.remove(hullFile)
            
        ### LABELS #############################################################################
        if(not ARGS.no_labels):
            
            # Compute labels
            if(ARGS.binary_labels):
                nc = 2
            else:
                nc = 3
            Y = np.zeros((mesh.nV, nc)) # [null, base, backbone] or [null, binding site]
            for atom in dna.get_atoms():
                # check if we include hydrogens
                if(not C["HYDROGENS"] and atom.element == 'H'):
                    continue
                aname = atom.name
                if(ARGS.binary_labels):
                    index = 1
                else:
                    if(regexes["DNA"]["BACKBONE"].search(aname)):
                        index = 2
                    else:
                        index = 1
                # get nearest vertices
                v, d = mesh.findVerticesInBall(atom.coord, C["DNA_MESH_DISTANCE_CUTOFF"])
                t = mesh.findTrianglesInBall(atom.coord, C["DNA_MESH_DISTANCE_CUTOFF"])
                if(len(v) > 0):
                    d = np.clip(1/(d+1e-5), 0.0, 2.0)
                    # check if atom-vertex segments intersect the mesh
                    ind = segmentsIntersectTriangles(
                        (np.tile(atom.coord, (len(v),1)), mesh.V[v]),
                        (mesh.V[t[:,0]], mesh.V[t[:,1]], mesh.V[t[:,2]])
                    ) 
                    v = v[ind]
                    d = d[ind]
                    # add distances to labels
                    Y[v, index] += d
            Y = np.argmax(Y, axis=1)
            
            if(len(dna) > 0):
                # Perform KNN smoothing on labels
                if(ARGS.smooth_labels):
                    #Yh1 = oneHotEncode(Y, nc)
                    #Yh2 = 2*Yh1 # count each vertex as two votes
                    #for i in range(mesh.nV):
                        #neighbors = mesh.findNeighbors(i)
                        #for n in neighbors:
                            #Yh2[i] += Yh1[n]
                    #Y = np.argmax(Yh2, axis=1)
                    tmesh = trimesh.Trimesh(vertices=mesh.V, faces=mesh.F, process=False, validate=False, vertex_attributes={'y': Y})
                    smoothLabels(tmesh, 'y', class_label=0)
                    smoothLabels(tmesh, 'y', class_label=1)
                
                if(ARGS.mask_labels):
                    # mask labels by distance criteria
                    Yn = np.argwhere(Y == 0).flatten()
                    if(ARGS.binary_labels):
                        Yp = (Y == 1)
                    else:
                        Yp = ((Y == 1) + (Y == 2))
                    Kn = cKDTree(mesh.V[Yn])
                    ind = [i for j in Kn.query_ball_point(mesh.V[Yp], C.get("MESH_BS_DISTANCE_CUTOFF", 5.0)) for i in j]
                    Y[Yn[ind]] = -1
        
        ### OUTPUT #############################################################################
        # Write features to disk
        if(not ARGS.no_features):
            if(not ARGS.no_potential):
                X = np.concatenate((Xm, Xa, Xh, Xp, Xe), axis=1)
                feature_names = features_m + features_a + features_h + features_p + features_e
            else:
                X = np.concatenate((Xm, Xa, Xh, Xp), axis=1)
                feature_names = features_m + features_a + features_h + features_p
            
            np.save(os.path.join(C['FEATURE_DATA_PATH'], "{}_vertex_features.npy".format(protein_id)), X, allow_pickle=False)
            FH = open(os.path.join(C['FEATURE_DATA_PATH'],"vertex_feature_names.dat"), "w")
            for i in range(len(feature_names)):
                FH.write("{:<2d} {}\n".format(i, feature_names[i]))
            FH.close()
            
            # vertex positions
            np.save(os.path.join(C['FEATURE_DATA_PATH'], "{}_vertex_positions.npy".format(protein_id)), mesh.V, allow_pickle=False)
            
            # faces indices
            np.save(os.path.join(C['FEATURE_DATA_PATH'], "{}_face_indices.npy".format(protein_id)), mesh.F, allow_pickle=False)
            
            # vertex normals
            if(mesh.Nv is not None):
                np.save(os.path.join(C['FEATURE_DATA_PATH'], "{}_vertex_normals.npy".format(protein_id)), mesh.Nv, allow_pickle=False)
        
        # Write labels to disk
        if(not ARGS.no_labels):
            np.save(os.path.join(C['FEATURE_DATA_PATH'], "{}_vertex_labels.npy".format(protein_id)), Y, allow_pickle=False)
        
        # Write mesh adjacency to disk
        if(not ARGS.no_adjacency):
            save_npz(os.path.join(C['FEATURE_DATA_PATH'], "{}_adj.npz".format(protein_id)), mesh.VA)
            print("Wrote mesh adjacency to disk for: {}".format(protein_id))
        
        ### CLEAN-UP ###########################################################################
        os.remove(pdb)
        os.remove("{}.pqr".format(protein_id))
    
    return 0

if __name__ == '__main__':
    main()

