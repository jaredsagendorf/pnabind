from .data import data

def getHBondAtoms(structure, hb_info=None, d_key="hb_donor", h_key="hb_donor", a_key="hb_acceptor", feature_value=1.0):
    # check for hydrogen bond info
    if hb_info is None:
        hb_info = data.hydrogen_bond_data
    
    # initialize atom keys
    for atom in structure.get_atoms():
        atom.xtra[d_key] = 0.0
        atom.xtra[h_key] = 0.0
        atom.xtra[a_key] = 0.0
    
    # loop over atoms and find donor/acceptors
    for atom in structure.get_atoms():
        aname = atom.name
        parent = atom.get_parent()
        parent_name = parent.get_resname()
        
        if parent_name in hb_info:
            # Donor Atoms
            if aname in hb_info[parent_name]['donors']:
                atom.xtra[d_key] = 1.0
                for h in hb_info[parent_name]['donors'][aname]['hydrogen_atoms']:
                    # loop over hydrogen atoms
                    if h in parent:
                        parent[h].xtra[h_key] = feature_value
            
            # Acceptor Atoms
            if aname in hb_info[parent_name]['acceptors']:
                if parent_name == 'HIS':
                    # check protonation state
                    if aname == 'ND1' and 'HD1' in parent:
                        continue 
                    if aname == 'NE2' and 'HE2' in parent:
                        continue
                atom.xtra[a_key] = feature_value
    
    return list(sorted(set([d_key, h_key, a_key])))
