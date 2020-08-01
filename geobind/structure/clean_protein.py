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

# built in modules
import logging

# third party modules
import numpy as np
from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer

# geobind modules
from .strip_hydrogens import stripHydrogens
from .data import data

class ResidueMutator(object):
    def __init__(self, tripeptides=None, components=None, standard_residues=None):
        """ The mutator object takes a non-standard residue or incomplete residue and modifies it
        """
        # get defaults if not provided
        if(standard_residues is None):
            standard_residues = data.standard_residues
        if(tripeptides is None):
            tripeptides = data.tripeptides
        if(components is None):
            components = data.chem_components
        self.components = components
        self.candidates = {}
        self.standard_residues = standard_residues
        self.imposer = SVDSuperimposer()
        self.parser = PDBParser(PERMISSIVE=1,QUIET=True)
        
        # build up candidate structures
        for fn in tripeptides:
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

def cleanProtein(structure, mutator=None, regexes=None, hydrogens=True):
    """ Perform any operations needed to modify the structure or sequence of a protein
    chain.
    """
    # set up needed objects
    if(regexes is None):
        regexes = data.regexes
    if(mutator is None):
        mutator = ResidueMutator(data.tripeptides, data.components)
    
    # remove hydrogens if requested
    if(not hydrogens):
        stripHydrogens(structure)
    
    # remove non-standard residues
    for chain in structure.get_chains():
        replace = []
        remove = []
        for residue in chain:
            resn = residue.get_resname().strip()
            if(mutator.standard(resn)):
                continue
            elif(resn == 'HOH' or resn == 'WAT'):
                remove.append(residue.get_id())
            elif(regexes["SOLVENT_COMPONENTS"].search(resn)):
                continue
            elif(mutator.modified(resn)):
                replace.append(residue.get_id())
            else:
                remove.append(residue.get_id())
        
        for rid in remove:
            logging.info("removed unrecognized residue: %s", chain[rid].get_resname())
            chain.detach_child(rid)
        
        for rid in replace:
            replacement = mutator.mutate(chain[rid])
            logging.info("replacing modified residue %s with %s", chain[rid].get_resname(), replacement.get_resname())
            chain.detach_child(rid)
            if(replacement):
                replacement.id = rid
                chain.add(replacement)
    
    return structure
