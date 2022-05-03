def getDSSP(model, PDBFileName, dssp_map=None, feature_name='secondary_structure', formatstr="{}({})"):
    try:
        from Bio.PDB.DSSP import DSSP
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The module 'Bio.PDB.DSSP' is required for this functionality!")
    
    if dssp_map is None:
        # map eight ss types to three
        dssp_map = {
            "H": formatstr.format(feature_name, "H"),
            "G": formatstr.format(feature_name, "H"),
            "I": formatstr.format(feature_name, "H"),
            "E": formatstr.format(feature_name, "S"),
            "B": formatstr.format(feature_name, "L"),
            "T": formatstr.format(feature_name, "L"),
            "S": formatstr.format(feature_name, "L"),
            "-": formatstr.format(feature_name, "L")
        }
    
    # run DSSP using the DSSP class from BioPython
    dssp = DSSP(model, PDBFileName)
    
    # store secondary structure in each atom property dict
    for chain in model:
        cid = chain.get_id()
        for residue in chain:
            rid = residue.get_id()
            dkey = (cid, rid)
            
            if dkey in dssp:
                ss = dssp_map[dssp[dkey][2]]
            else:
                ss = dssp_map['-']
            
            for atom in residue:
                atom.xtra[ss] = 1.0
    
    return list(set(dssp_map.values()))
