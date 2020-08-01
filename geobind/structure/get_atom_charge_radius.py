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
    
    # Strip any existing hydrogens - add new ones with PDB2PQR
    stripHydrogens(structure)
    
    # Write chain to temp file
    pdbFile = "{}_temp.pdb".format(prefix)
    pqrFile = "{}.pqr".format(prefix)
    structure.save(pdbFile)
    
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
    if(not os.path.exists(pqrFile)):
        raise FileNotFoundError("No PQR file was produced ({}). Try manually running PDB2PQR on the pbdfile file '{}' and verify output.".format(pqrFile, pdbFile))
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
            vdw = min_radius # 0 radius atoms causes issues - set to a minimum of 0.6
        if(rid in model[cid]):
            if(atm in model[cid][rid]):
                model[cid][rid][atm].xtra["charge"] = crg 
                model[cid][rid][atm].xtra["radius"] = vdw
    model = StructureData(model, name=prefix)
    
    if(not hydrogens):
        # remove hydrogens
        stripHydrogens(model)
    
    # clean up
    os.remove(pdbFile)
    if(not keepPQR):
        os.remove(pqrFile)
    
    if(keepPQR):
        return model, pqrFile
    else:
        return model
