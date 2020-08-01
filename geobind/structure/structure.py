# builtin modules
import os
import logging

# third party modules
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.Entity import Entity
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model #Chain, Residue, Atom
import numpy as np

# geobind modules
from .get_atom_kdtree import getAtomKDTree
from .get_surface_residues import getSurfaceResidues

class StructureData(object):    
    def __init__(self, structure, name='structure', path='.'):
        if isinstance(structure, str):
            file_type = (str(structure).split('.')[-1]).lower()
            if file_type in ('pdb', 'ent'):
                # load a PDB file 
                __parser = PDBParser(PERMISSIVE=1, QUIET=True)
                self.structure = __parser.get_structure(name, os.path.join(path, structure))
            elif file_type == 'cif':
                # load MMCIF file
                __parser = MMCIFParser(QUIET=True)
                self.structure = __parser.get_structure(name, os.path.join(path, structure))
            else:
                raise ValueError("Unknown filetype for structure file name: {}".format(structure))
        elif isinstance(structure, Entity):
            # use structure as-is
            self.structure = structure
        else:
            raise ValueError("Unknown type for input argument 'structure': {}".format(str(structure)))
        
        # properties
        self.name = name
        
        # cachable properties
        self.cache = {}
        self._atom_KDTree = None
        self._atom_list = None
        self._surface_residues = None
        
    def __getitem__(self, key):
        return self.structure[key]
    
    def __iter__(self):
        for item in self.structure:
            yield item
    
    @classmethod
    def slice(cls, obj, selection, name='slice'):
        """Create a new Structure object 'S2' from a slice of the current one, 'S1'. <selection> 
        defines which  descendents 'S1' will be stored in 'S2'."""
        
        ### ONLY SUPPORTS SLICING UP TO MODEL->CHAIN LEVEL
        ent = Structure(name) # Biopython structure object
        for mid in selection:
            ent.add(Model(mid))
            for cid in selection[mid]:
                ent[mid].add(obj[mid][cid])
        
        return cls(ent, name=name)
    
    @property
    def atom_list(self):
        if "atom_list" not in self.cache:
            self.cache["atom_list"] = [atom for atom in self.get_atoms()]
        return self.cache["atom_list"]
    
    @property
    def atom_KDTree(self):
        if "kdtree" not in self.cache:
            self.cache["kdtree"] = getAtomKDTree(self.atom_list, engine='biopython')
        return self.cache["kdtree"]
    
    def add(self, item):
        self.structure.add(item)
    
    def get_level(self):
        return self.structure.get_level()
    
    def get_models(self):
        if self.get_level() in ('S'):
            return self.structure.get_models()
        else:
            raise AttributeError("This method is only defined for 'Structure' level objects!")
    
    def get_chains(self):
        if self.get_level() in ('S', 'M'):
            return self.structure.get_chains()
        else:
            raise AttributeError("This method is only defined for 'Structure' and 'Model' level objects!")
    
    def get_residues(self):
        if self.get_level() in ('S', 'M', 'C'):
            return self.structure.get_residues()
        else:
            raise AttributeError("This method is only defined for 'Structure', 'Model' and 'Chain' level objects!")
    
    def get_atoms(self):
        if self.get_level() == 'A':
           return [self.structure] 
        else:
            return self.structure.get_atoms()
    
    def get_residue(self, res_id, chain_id=None, mi=0):
        if self.get_level() == 'S':
            return self.structure[mi][chain_id][res_id]
        elif self.get_level() == 'M':
            return self.structure[chain_id][res_id]
        elif self.get_level() == 'C':
            return self.structure[res_id]
        else:
            return None
        
    def detach_child(item):
        self.structure.detach_child(item)
    
    def get_surface_residues(self, hydrogens=False, area_key='sesa'):
        if(self._surface_residues is not None):
            return self._surface_residues
        else:
            self._surface_residues = getSurfaceResidues(self.structure, area_key=area_key, hydrogens=hydrogens)
            return self._surface_residues
    
    def getNearestNeighbor(self, atom, cutoff=3.0, eps=1e-5, hydrogens=True):
        neighbors = self.atom_KDTree.search(atom.coord, cutoff)
        mindist = 99999
        nn = None
        for n in neighbors:
            if (n.element == 'H') and (not hydrogens):
                continue
            dist = np.linalg.norm(atom.coord - n.coord)
            if dist < mindist and dist > eps:
                mindist = dist
                nn = n
        
        return nn
    
    def save(self, outfile=None):
        __io = PDBIO()
        # write structure to file
        if(outfile is None):
            outfile = self.name + ".pdb"
        logging.debug("Saving pdb file: %s", outfile)
        __io.set_structure(self.structure)
        __io.save(outfile)
        
        return outfile
