import numpy as np

def getCOM(residue, hydrogens=True):
    com = np.zeros(3)
    count = 0.0
    for atom in residue:
        if atom.element == 'H' and not hydrogens:
            continue
        com += atom.get_coord()
        count += 1.0
    return com/count

def getResidueDistance(res1, res2, atm1=None, atm2=None, hydrogens=True):
    distances = []
    for a1 in res1:
        if a1.element == 'H' and not hydrogens:
            continue
        distances.append([])
        for a2 in res2:
            if a2.element == 'H' and not hydrogens:
                continue
            d = a1-a2
            distances[-1].append(d)
    
    # center of mass
    cm1 = getCOM(res1)
    cm2 = getCOM(res2)
    
    distance_measures = {
        "min": np.min(distances),
        "mean_nn": np.min(distances, axis=1).mean(),
        "com": np.linalg.norm(cm1-cm2),
    }
    
    if atm1 and atm2:
       distance_measures["{}-{}".format(atm1, atm2)] = res1[atm1] - res2[atm2] 
    
    return distance_measures

def getResidueCADistances(residues):
    atoms = [residue['CA'] for residue in residues]
    distances = []
    
    N = len(atoms)
    for i in range(N):
        for j in range(i+1, N):
            distances.append(atoms[i] - atoms[j])
    
    return distances
