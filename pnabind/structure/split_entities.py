# pnabind modules
from .data import data as D


def isAA(resname, regexes):
    if regexes['PROTEIN']['STANDARD_RESIDUES'].search(resname):
        return True
    
    if resname in D.chem_components:
        ccd = D.chem_components[resname]
        if '_chem_comp.mon_nstd_parent_comp_id' in ccd:
            parent = ccd['_chem_comp.mon_nstd_parent_comp_id']
            if regexes['PROTEIN']['STANDARD_RESIDUES'].search(parent):
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def splitEntities(structure, regexes=None, atom_mapper=None, mi=0, remove_peptides=True, min_chain_length=None):
    """Docstring"""
    if regexes is None:
        regexes = D.regexes
    
    pro = []
    lig = []
    for chain in structure[mi]:
        for residue in chain:
            resname = residue.get_resname().strip()
            hetflag = residue.get_id()[0]
            aa = isAA(resname, regexes)
            if aa:
                if remove_peptides and hetflag != ' ':
                    continue
                if (min_chain_length is not None) and (len(chain) < min_chain_length):
                    continue
                pro.append(residue.get_full_id())
            else:
                if atom_mapper is not None:
                    if atom_mapper.testResidue(residue):
                        lig.append(residue.get_full_id())
                else:
                    lig.append(residue.get_full_id())
    
    return structure.slice(structure, pro, name='{}_protein'.format(structure.name)), structure.slice(structure, lig, name='{}_ligand'.format(structure.name))
