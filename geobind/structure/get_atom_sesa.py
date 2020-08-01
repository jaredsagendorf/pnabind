# standard modules
import os

# geobind modules
from geobind.mesh import runMSMS

def getAtomSESA(structure, prefix, clean=True, hydrogens=False):
    atoms = structure.atom_list
    
    # run MSMS
    af = runMSMS(atoms, prefix, '.', area_only=True, hydrogens=hydrogens)
    
    # read in area file
    SE = open(af)
    SE.readline()
    count = 0
    for i in range(len(atoms)):
        if((not hydrogens) and atoms[i].element == 'H'):
            continue
        sesa = float(SE.readline().strip().split()[1])
        atoms[i].xtra['sesa'] = sesa
        count += 1
    
    # clean up
    if(clean):
        os.remove(af)
