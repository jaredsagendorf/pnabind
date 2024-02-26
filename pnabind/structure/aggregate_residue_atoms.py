import numpy as np

def aggregateResidueAtoms(atoms, atom_key, residue_key=None, reduce_method="sum", impute_atom=False, atom_impute_value=0.0, residue_impute_value=0.0):
    
    # reduction operations
    if reduce_method == 'sum':
        reduce_fn = np.sum
    elif reduce_method == 'mean':
        reduce_fn = np.mean
    elif reduce_method == 'max':
        reduce_fn = np.max
    elif reduce_method == 'min':
        reduce_fn = np.min
    
    if residue_key is None:
        residue_key = atom_key
    
    # map
    residues = set()
    for atom in atoms:
        # create corresponding residue entry if needed
        residue = atom.get_parent()
        
        if not hasattr(residue, 'xtra'):
            # create dict to store info
            residue.xtra = {}
        
        if residue_key not in residue.xtra:
            # add new key
            residue.xtra[residue_key] = []
        
        # get atom value
        if atom_key not in atom.xtra:
            if impute_atom:
                val = atom_impute_value
            else:
                continue
        else:
            val = atom.xtra[atom_key]
        residue.xtra[residue_key].append(val)
        
        residues.add(residue)
    
    # reduce
    for residue in residues:
        if len(residue.xtra[residue_key]) == 0:
            residue.xtra[residue_key] = residue_impute_value
        else:
            residue.xtra[residue_key] = reduce_fn(residue.xtra[residue_key])
    
    return list(residues)
