# builtin modules
import os

# third party modules
import freesasa

# geobind modules
from .data import data

class Radius(freesasa.Classifier):
    def initialize(self, components=None, radii=None):
        if(components is None):
            components = data.chem_components
            
        if(radii is None):
            radii = data.vdw_radii
            
        self.components = components
        self.radii = radii
    
    def radius(self, residueName, atomName):
        rName = residueName.strip()
        aName = atomName.strip()
        if(rName in self.radii):
            # standard Residue
            if(aName in self.radii[rName]):
                return self.radii[rName][aName]
            else:
                return self.radii['element'][self.getElement(rName, atomName)]
        elif(rName in self.components):
            # non-standard known residue
            parent = self.components[rName]['_chem_comp.mon_nstd_parent_comp_id']
            if(parent in self.radii and aName in self.radii[parent]):
                return self.radii[parent][aName]
            else:
                return self.radii['element'][self.getElement(rName, atomName)]
        else:
            # unknown residue - make best guess for atom element
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

def getFreeSASAStructureFromModel(structure, classifier=None):
    outFile = "gsfm.temp.pdb"
    structure.save(outFile)
    
    if(classifier is not None):
        freesasa_structure = freesasa.Structure(outFile, classifier=classifier)
    else:
        freesasa_structure = freesasa.Structure(outFile)
    
    if(os.access(outFile, os.R_OK)):
        os.remove(outFile)
    
    return freesasa_structure
    
def getAtomSASA(structure, classifier=None, probe_radius=1.4, mi=0, feature_name="sasa", **kwargs):
    
    if classifier is None:
        # initialize new classifier
        classifier = Radius(**kwargs)
    
    freesasa_structure = getFreeSASAStructureFromModel(structure, classifier=classifier)
    SASA = freesasa.calc(freesasa_structure, freesasa.Parameters({"probe-radius": probe_radius}))
    
    # get atom SASA
    N = structure.nAtoms()
    for i in range(N):
        sasa = SASA.atomArea(i)
        resi = freesasa_structure.residueNumber(i).strip()
        cid = freesasa_structure.chainLabel(i).strip()
        if(resi[-1].isdigit()):
            ins = " "
        else:
            ins = resi[-1]
            resi = resi[:-1]
        aname = structure.atomName(i).strip()
        structure[mi][cid][(' ', int(resi), ins)][aname].xtra[feature_name] = sasa
    
    return feature_name
