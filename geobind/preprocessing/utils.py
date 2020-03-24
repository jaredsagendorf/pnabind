import os
import json
import re
import freesasa
from Bio.PDB import PDBIO
import numpy as np

RESIDUES = [
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLU',
    'GLN',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL'
]

PLANAR_RESIDUES = [
    "ARG",
    "PHE",
    "TYR",
    "TRP",
    "HIS",
    "ASN",
    "ASP",
    "GLN",
    "GLU"
]

NUCLEOTIDES = [
    "DA",
    "DC",
    "DG",
    "DT"
]

DNA_MTY = [
    ["bs", "sg", "wg"],
    ["sr"],
    ["pp"]
]

BASE_MTY = set(["bs", "sg", "wg"])

long_to_short = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "DA": "A",
    "DC": "C",
    "DG": "G",
    "DT": "T",
    "DI": "I",
    "5CM": "m",
    "DU": "U"
}

class Scaler(object):
    def __init__(self, data):
        self.min = data.min()
        self.max = data.max()
    
    def __call__(self, value):
        return (value-self.min)/(self.max-self.min)

class Radius(freesasa.Classifier):
    def initialize(self, DATA_PATH, COMPONENTS, fileName='vdw-radii.json'):
        self.components = COMPONENTS
        with open(os.path.join(DATA_PATH, fileName)) as FILE:
            self.radii = json.load(FILE)
    
    def radius(self, residueName, atomName):
        rName = residueName.strip()
        aName = atomName.strip()
        if(rName in self.radii):
            # Standard Residue
            if(aName in self.radii[rName]):
                return self.radii[rName][aName]
            else:
                return self.radii['element'][self.getElement(rName, atomName)]
        elif(rName in self.components):
            # Non-standard known residue
            parent = self.components[rName]['_chem_comp.mon_nstd_parent_comp_id']
            if(parent in self.radii and aName in self.radii[parent]):
                return self.radii[parent][aName]
            else:
                return self.radii['element'][self.getElement(rName, atomName)]
        else:
            # Unknown residue - make best guess for atom element
            print("Unknown residue: {}".format(rName))
            return self.radii['element'][self.guessElement(atomName)]
    
    def classify(self, residueName, atomName):
        return "atom"
    
    def getElement(self, residueName, atomName):
        aName = atomName.strip()
        if(residueName in self.components):
            try:
                index = self.components[residueName]['_chem_comp_atom.atom_id'].index(aName)
                return self.components[residueName]['_chem_comp_atom.type_symbol'][index]
            except:
                return self.guessElement(atomName)
        else:
            return self.guessElement(atomName)
    
    def guessElement(self, atomName):
        """Tries to guess element from atom name if not recognised."""
        print("Got :{}".format(atomName))
        name = atomName.strip()
        if name.capitalize() not in self.radii["element"]:
            # Inorganic elements have their name shifted left by one position
            #  (is a convention in PDB, but not part of the standard).
            # isdigit() check on last two characters to avoid mis-assignment of
            # hydrogens atoms (GLN HE21 for example)
            if atomName[0].isalpha() and not (atomName[2:].isdigit() or atomName[2:] == "''"):
                putative_element = name
            else:
                # Hs may have digit in first position
                if name[0].isdigit():
                    putative_element = name[1]
                else:
                    putative_element = name[0]
        
            if putative_element.capitalize() in self.radii["element"]:
                element = putative_element
            else:
                element = ""
            return element
        else:
            return name

def compileRegexes(obj):
    # compile regexes loaded from JSON files
    objType = type(obj)
    if(objType is dict):
        for key in obj:
            if(type(obj[key]) is str):
                obj[key] = re.compile(obj[key])
            else:
                compileRegexes(obj[key])
    elif(objType is list):
        for i in range(len(obj)):
            if(type(obj[i]) is str):
                obj[i] = re.compile(obj[i])
            else:
                compileRegexes(obj[i])

def getStructureFromModel(model, classifier=None):
    outFile = "gsfm.temp.pdb"
    io = PDBIO()
    io.set_structure(model)
    io.save(outFile)
    
    if(classifier is not None):
        structure = freesasa.Structure(outFile, classifier=classifier)
    else:
        structure = freesasa.Structure(outFile)
    
    if(os.access(outFile, os.R_OK)):
        os.remove(outFile)
    
    return structure

def read_interface_list(fileName):
    INTERFACES = {}
    intFile = open(fileName)
    for line in intFile:
        line = line.strip().split(',')
        pdbid, interface, mnum = line[0].split(':')
        dna_id, pro_id = interface.split('_')
        pro_id = pro_id[0]
        mnum = int(mnum)
        
        if(pdbid not in INTERFACES):
            INTERFACES[pdbid] = {} # keyed by entity_chain
        if(interface not in INTERFACES[pdbid]):
            INTERFACES[pdbid][interface] = {
                "dna_id": dna_id,
                "pro_id": pro_id,
                "mnum": mnum
            }
    intFile.close()
    return INTERFACES

def iterateInterfaces(INTERFACES, flat=False, datadir="."):
    for pdbid in INTERFACES:
        # Get the structure data
        if(flat):
            with open(os.path.join(datadir, "{}.json".format(pdbid))) as FH:
                DATA = json.load(FH)
        else:
            with open(os.path.join(datadir, pdbid[-1], "{}.json".format(pdbid))) as FH:
                DATA = json.load(FH)
        
        if("error" in DATA):
            print("{}: JSON data missing".format(pdbid))
            continue
        
        # Get a residue map
        RESIDUE_MAP = {}
        for res in DATA["protein"]["residues"]:
            RESIDUE_MAP[res["id"]] = res
        
        # Get the interfaces in the structure
        for int_id in INTERFACES[pdbid]:
            interface = None
            interface_stats = None
            dna_id = INTERFACES[pdbid][int_id]["dna_id"]
            pro_id = INTERFACES[pdbid][int_id]["pro_id"]
            mnum = INTERFACES[pdbid][int_id]["mnum"]
            # find interface
            for intf in DATA["interfaces"]["models"][mnum]:
                if(intf["dna_entity_id"] == dna_id and pro_id in intf["protein_chains"]):
                    interface = intf
                    break
            if(interface is None):
                print(pdbid, dna_id, mnum)
                exit(0)
            
            # find interface stats
            for intf_stats in interface["interface_features"]:
                if(intf_stats["protein_chain_id"] == pro_id):
                    interface_stats = intf_stats
                    break
            if(interface_stats is None):
                print(pdbid, dna_id, pro_id, mnum)
                exit(0)
            
            yield interface, interface_stats, RESIDUE_MAP, dna_id, pro_id, mnum

def stripHydrogens(structure):
    """Strip all hydrogen atoms from the given model.
    
    Parameters
    ----------
    model: BioPython MODEL object
        The model to be stripped of hydrogens.
    """
    for residue in structure.get_residues():
        rm = []
        for atom in residue:
            if(atom.element == 'H'):
                rm.append(atom.get_id())
        for aid in rm:
            residue.detach_child(aid)
