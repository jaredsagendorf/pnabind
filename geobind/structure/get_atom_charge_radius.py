# builtin modules
import subprocess
import os

# third party modules
from Bio.PDB import PDBParser

# geobind modules
from .strip_hydrogens import stripHydrogens
from .structure import StructureData

def getAtomChargeRadius(structure, prefix='structure', hydrogens=True, keepPQR=False, usePDB2PQR=True, min_radius=0.6):
    """ Assign atomic parameters to every atom in a protein chain. Values are stored in atom.xtra 
    dictionary. The following keys are used
    ------------------------------------------------------------------------------------------------
    radius - van der Waals radius
    charge - atomic effective charge
    
    ARGUMENTS:
        model - a geobind Structure object
    """
    pass
