# standard modules
import os

# geobind modules
from geobind.mesh import runMSMS
from .structure import StructureData

def getAtomSESA(atoms, prefix='area', clean=True, hydrogens=True, quiet=True, key='sesa'):
    if isinstance(atoms, StructureData):
        atoms = atoms.atom_list
    
    # run MSMS
    af = runMSMS(atoms, prefix, '.', area_only=True, hydrogens=hydrogens, quiet=quiet)
    
    # read in area file
    SE = open(af)
    SE.readline()
    count = 0
    for i in range(len(atoms)):
        if (not hydrogens) and atoms[i].element == 'H':
            continue
        sesa = float(SE.readline().strip().split()[1])
        atoms[i].xtra[key] = sesa
        count += 1
    
    # clean up
    if clean:
        os.remove(af)
