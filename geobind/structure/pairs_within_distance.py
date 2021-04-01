from .get_atom_kdtree import getAtomKDTree

def pairsWithinDistance(struct1, struct2, distance=4.0, hydrogens=False, id_format='dnaprodb'):
    struct1_residue_ids = set()
    struct2_residue_ids = set()
    
    if hydrogens:
        kdt = struct1.atom_KDTree
    else:
        alist = []
        for atom in struct1.get_atoms():
            if atom.element == "H":
                continue
            alist.append(atom)
        kdt = getAtomKDTree(alist, engine="biopython")
    
    for residue2 in struct2.get_residues():
        for atom2 in residue2:
            if (not hydrogens) and atom2.element == 'H':
                continue
            
            residues1 = kdt.search(atom2.coord, distance, level='R')
            if len(residues1) > 0:
                struct2_residue_ids.add(residue2.get_full_id())
                struct1_residue_ids.update([r.get_full_id() for r in residues1])
    
    if id_format == 'dnaprodb':
        struct1_residue_ids = ["{}.{}.{}".format(_[2], _[3][1], _[3][2]) for _ in  struct1_residue_ids]
        struct2_residue_ids = ["{}.{}.{}".format(_[2], _[3][1], _[3][2]) for _ in  struct2_residue_ids]
    else:
        struct1_residue_ids = list(struct1_residue_ids)
        struct2_residue_ids = list(struct2_residue_ids)
    
    return struct1_residue_ids, struct2_residue_ids
