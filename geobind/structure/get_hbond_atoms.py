from .data import data

def getHBondAtoms(structure, hb_info=None, d_key="hb_donor", h_key="hb_donor", a_key="hb_acceptor"):
    
    atoms = structure.get_atoms()
    
    # check for hydrogen bond info
    if(hb_info is None):
        hb_info = data.hydrogen_bond_data
    
    # loop over atoms
    for atom in atoms:
        aname = atom.name
        parent = atom.get_parent()
        parent_name = parent.get_resname()
        
        if(parent_name in hb_info):
            # Donor Atoms
            if(aname in hb_info[parent_name]['donors']):
                atom.xtra[d_key] = 1.0
                for h in hb_info[parent_name]['donors'][aname]['hydrogen_atoms']:
                    # loop over hydrogen atoms
                    if(h in parent):
                        parent[h].xtra[h_key] = 1.0
            
            # Acceptor Atoms
            if(aname in hb_info[parent_name]['acceptors']):
                if(parent_name == 'HIS'):
                    # check protonation state
                    if(aname == 'ND1' and 'HD1' in parent):
                        continue 
                    if(aname == 'NE2' and 'HE2' in parent):
                        continue
                atom.xtra[a_key] = 1.0
    
    return list(set([d_key, h_key, a_key]))
