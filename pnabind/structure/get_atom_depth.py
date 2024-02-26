
def getAtomDepth(atoms, vertices, feature_name="depth"):
    try:
        from Bio.PDB.ResidueDepth import min_dist
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The module 'Bio.PDB.ResidueDepth' is required for this functionality!")
    
    if not isinstance(atoms, list):
        atoms = atoms.get_atoms()
    
    for atom in atoms:
        coord = atom.coord
        dist = min_dist(vertices, coord)
        
        atom.xtra[feature_name] = dist
    
    
    return feature_name
