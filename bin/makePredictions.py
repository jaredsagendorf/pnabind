import os
import sys
import argparse
import logging
import numpy as np
from sklearn.metrics import jaccard_score, balanced_accuracy_score, confusion_matrix, auc, f1_score, fbeta_score
import trimesh
from glob import glob
import math

from geobind import vertexLabelsToResidueLabels
from geobind.nn.metrics import chooseBinaryThreshold, reportMetrics, auroc, auprc
from geobind.mesh import smoothMeshLabels
from geobind.nn.metrics import meshLabelSmoothness
from geobind.structure import mapPointFeaturesToStructure #mapVertexProbabilitiesToStructure
from geobind.structure import getAtomKDTree, StructureData, getAtomSESA, cleanProtein
from geobind.structure.data import data as D

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("path",
                help="directory containing .npz files")
arg_parser.add_argument("--pdb_dir", default=".",
                help="directory containing PDB files")
arg_parser.add_argument("--min_area", type=float,
                help="minimum cluster area")
arg_parser.add_argument("--merge_dist", type=float,
                help="label cluster merge distance")
arg_parser.add_argument("-t", "--threshold", type=float, default=0.5,
                help="probability threshold for evaluating metrics")
arg_parser.add_argument("--metric_opt", default="ba",
                choices=["ba", "miou", "smoothness", "f1", "f2"],
                help="metric to optimize threshold with")
arg_parser.add_argument("--metrics_eval", nargs="+", default=["miou", "ba", "auc"],
                choices=["auc", "ba", "miou", "smoothness", "f1", "f2", "standard", "sp", "fpr"],
                help="metrics to evaluate")
arg_parser.add_argument("--no_print_individual", action="store_true",
                help="don't print metrics for individual structures")
arg_parser.add_argument("--training_data",
                help="use training data to optimize threshold")
arg_parser.add_argument("--no_residue_predictions", action="store_true", 
                help="skip residue predictions")
arg_parser.add_argument("--write_predictions", action="store_true", 
                help="write out predictions to file")
arg_parser.add_argument("--csv_file", 
                help="write output to CSV file")
ARGS = arg_parser.parse_args()

# set up logging
log_format = '%(levelname)s:    %(message)s'
filename = None
log_level = logging.INFO
logging.basicConfig(format=log_format, filename=filename, level=log_level)

def getProtein(structure, regexes, mi=0):
    """Docstring"""
    pro = []
    for chain in structure[mi]:
        for residue in chain:
            resname = residue.get_resname()
            if regexes['PROTEIN']['STANDARD_RESIDUES'].search(resname):
                pro.append(residue.get_full_id())
    
    return structure.slice(structure, pro, 'protein')

def reduceDict(d, fn=None):
    for key in d:
        if isinstance(d[key], dict):
            reduceDict(d[key], fn=fn)
        else:
            d[key] = fn(d[key])

def catDicts(d1, d2):
    for key in d2:
        if isinstance(d2[key], dict):
            if key not in d1:
                d1[key] = {}
            catDicts(d1[key], d2[key])
        else:
            if key not in d1:
                d1[key] = []
            d1[key].append(d2[key])

def getAUC(Y, P, mesh=None, mask=None, threshold=None):
    
    def getAUROC(fpr, tpr):
        _roc = 0
        d = -np.diff(fpr)
        di = np.argwhere(d < 0).flatten() + 1
        si = 0
        for i in range(len(di)):
            ei = di[i]
            if ei > (si + 1):
                _roc += auc(fpr[si:ei], tpr[si:ei])
            else:
                _roc += abs(fpr[si] - fpr[si+1])*(tpr[si+1] + tpr[si])/2
            si = ei
        ei = len(fpr)
        _roc += auc(fpr[si:ei], tpr[si:ei])
        
        return _roc
        
    if mask is None:
        mask = np.array([True]*len(Y))
    
    if threshold:
        # get mesh info
        E = mesh.edges_unique
        area_faces = mesh.area_faces
        
        # compute threshold metrics
        tpr = []
        fpr = []
        pre = []
        
        # sample threshold values uniformly
        N = 50
        t = np.linspace(0, 1, N)
        for i in range(N):
            Ypr = (P[:,1] >= t[i]).astype(int)
            Ypr = smoothMeshLabels(E, Ypr, 2, threshold=threshold, area_faces=area_faces, faces=F)
            tn, fp, fn, tp = confusion_matrix(Y[mask], Ypr[mask]).ravel()
            
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
            if tp + fp > 0:
                pre.append(tp/(tp+fp))
            else:
                pre.append(0)
        
        # auroc
        roc = -np.trapz(y=tpr, x=fpr)#getAUROC(fpr, tpr)
        
        # auprc
        tpr.reverse()
        pre.reverse()
        prc = (np.diff(tpr)*np.array(pre[1:])).sum() #auc(tpr, pre)
    else:
        roc = auroc(Y[mask], P[mask])
        prc = auprc(Y[mask], P[mask])
    
    return roc, prc

def confusionMatrixMetircs(y1, y2):
    tn, fp, fn, tp = confusion_matrix(y1, y2).ravel()
    metrics = {}
    
    if (tp + fn) > 0:
        TPR = tp/(tp + fn)
    else:
        TPR = float('nan')
    
    if (tn + fp) > 0:
        TNR = tn/(tn + fp)
    else:
        TNR = float('nan')
    metrics["ac"] = (tp + tn)/(tp + tn + fn + fp)
    metrics["ba"] = (TPR + TNR)/2
    metrics["re"] = TPR
    metrics["sp"] = TNR
    
    # precision score
    if (tp + fp) > 0:
        metrics["pr"] = tp/(tp + fp)
    else:
        metrics["pr"] = float('nan')
    
    # f1 score
    b2 = 4.0
    if (metrics["re"] + metrics["pr"]) == 0 or math.isnan(metrics["re"] + metrics["pr"]):
        metrics["f1"] = float('nan')
        metrics["f2"] = float('nan')
    else:
        metrics["f1"] = 2*metrics["re"]*metrics["pr"]/(metrics["re"] + metrics["pr"])
        metrics["f2"] = (1+b2)*metrics["pr"]*metrics["re"]/(b2*metrics["pr"] + metrics["re"])
    
    return metrics

def specificity(y1, y2):
    ni = (y1 == 0)
    tn = (y2[ni] == 0).sum()
    
    return tn/(ni.sum())

def fpr(y1, y2):
    pi = (y2 == 1)
    fp = (y1[pi] == 0).sum()
    
    return fp/( (y1 == 0).sum() )

def printCSV(FH, metrics, label="", header=True):
    if FH is not None:
        fields = [label]
        headers = ["label"]
        for k1 in sorted(metrics.keys()):
            for k2 in sorted(metrics[k1].keys()):
                headers.append(k2)
                fields.append("{:.3f}".format(metrics[k1][k2]))
        
        if header:
            FH.write(','.join(headers) + '\n')
        FH.write(','.join(fields) + '\n')

# def getResidueLabels(atoms, mesh, Y, kdt):
    # bs_areas = np.zeros_like(Y, dtype=np.float32)
    # np.add.at(bs_areas, mesh.faces[:, 0], mesh.area_faces/3)
    # np.add.at(bs_areas, mesh.faces[:, 1], mesh.area_faces/3)
    # np.add.at(bs_areas, mesh.faces[:, 2], mesh.area_faces/3)
    
    # ni = (Y == 0)
    # bs_areas[ni] = 0
    
    # mapPointFeaturesToStructure(mesh.vertices, atoms, bs_areas, 'bs_sesa', kdtree=kdt)
    
    # residue_dict = {}
    # # aggregate over atom areas
    # for atom in atoms:
        # residue = atom.get_parent()
        # residue_id = residue.get_full_id()
        # if residue_id not in residue_dict:
            # residue_dict[residue_id] = {
                # 'sesa': 0,
                # 'bs_sesa': 0
            # }
        # residue_dict[residue_id]['sesa'] += atom.xtra['sesa']
        # residue_dict[residue_id]['bs_sesa'] += atom.xtra['bs_sesa']
    
    # for key in residue_dict:
        # if residue_dict[key]['bs_sesa'] > 5.0:
            # residue_dict[key]['label'] = 1.0
        # else:
            # residue_dict[key]['label'] = 0.0
    
    #return residue_dict

def smoothResiduePredictions(protein, residue_dict):
    pass

metric_fns = {
    "miou": lambda y1, y2: jaccard_score(y1, y2, average='weighted'),
    "ba": lambda y1, y2: balanced_accuracy_score(y1, y2),
    "smoothness": lambda y, E: meshLabelSmoothness(y, E),
    "f1": lambda y1, y2: f1_score(y1, y2, average="binary"),
    "f2": lambda y1, y2: fbeta_score(y1, y2, beta=2.0, average="binary"),
    "standard": lambda y1, y2: confusionMatrixMetircs(y1, y2),
    "sp": lambda y1, y2: specificity(y1, y2),
    "fpr": lambda y1, y2: fpr(y1, y2)
}

if ARGS.training_data:
    training = np.load(ARGS.training_data)
    mask = training['Y'] >= 0
    Ytr = training['Y'][mask]
    Ptr = training['P'][mask]
    if Ptr.ndim > 2:
        Ptr = Ptr.mean(axis=-1) # reduce ensemble
    
    t, m = chooseBinaryThreshold(Ytr, Ptr[:,1], metric_fns[ARGS.metric_opt])
    ARGS.threshold = t
    print("Optimized threshold ({}): {:.2f} score: {:.3f}".format(ARGS.metric_opt, t, m))

if ARGS.csv_file:
    CSV = open(ARGS.csv_file, "w")
else:
    CSV = None

header = True
file_names = glob(os.path.join(ARGS.path, "*.npz"))
lw = max([len(_)-len("./_predict.npz") for _ in file_names] + [len("Protein Identifier")])
metrics_all = []
for line in file_names:
    # load data arrays
    data = np.load(line)
    Y = data['Y']
    P = data['Pm']
    mask = (Y >= 0)
    V = data['V']
    F = data['F']
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    
    # Get predictions for given threshold
    Ypr = (P[:,1] >= ARGS.threshold).astype(int)
    
    if ARGS.min_area:
        # Apply smoothing
        E = mesh.edges_unique
        Ypo = smoothMeshLabels(E, Ypr.copy(), 2, 
                threshold=ARGS.min_area,
                merge_nearby_clusters=True,
                merge_distance=ARGS.merge_dist,
                vertices=V,
                faces=F,
                area_faces=mesh.area_faces
        )
    
    if not ARGS.no_residue_predictions:
        # Get stucture based pooling
        
        # apply same processing as on training data
        fname = line.replace("_predict.npz", "")
        fname = fname.replace("./", "")
        structure = StructureData(fname+'.pdb', name=fname, path=ARGS.pdb_dir)
        protein = getProtein(structure, D.regexes)
        
        # Clean the protein entity
        protein, pqr = cleanProtein(protein, hydrogens=True, quiet=False)
        
        # get atoms with sesa > 0
        getAtomSESA(protein)
        atoms = list(filter(lambda atom: 'sesa' in atom.xtra and atom.xtra['sesa'] > 0, [atom for atom in protein.get_atoms()]))
        kdt = getAtomKDTree(atoms)
        
        # # map probabilities
        # vertex_areas = np.zeros_like(Y, dtype=np.float32)
        # np.add.at(vertex_areas, F[:, 0], mesh.area_faces/3)
        # np.add.at(vertex_areas, F[:, 1], mesh.area_faces/3)
        # np.add.at(vertex_areas, F[:, 2], mesh.area_faces/3)
        
        
        # Ynm = Y.copy()
        # Ynm[~mask] = 0 # remove mask
        # map_gt = mapVertexProbabilitiesToStructure(V, atoms, Ynm, 2, kdtree=kdt, level=ARGS.level, reduce_method='max', vertex_weights=vertex_areas)
        # map_p = mapVertexProbabilitiesToStructure(V, atoms, P, 2, kdtree=kdt, level=ARGS.level, reduce_method='max', vertex_weights=vertex_areas)
        # map_pr = mapVertexProbabilitiesToStructure(V, atoms, Ypr, 2, kdtree=kdt, level=ARGS.level, reduce_method='max', vertex_weights=vertex_areas)
        # map_po = mapVertexProbabilitiesToStructure(V, atoms, Ypo, 2, kdtree=kdt, level=ARGS.level, reduce_method='max', vertex_weights=vertex_areas)
        
        # Ys = []
        # Yspr = []
        # Yspo = []
        # Ps = []
        # ids = []
        # for key in map_gt:
            # Ys.append(map_gt[key][1])
            # Yspr.append(map_pr[key][1])
            # Yspo.append(map_pr[key][1])
            # Ps.append(map_p[key])
            # ids.append(key)
        
        # Ys = (np.array(Ys) >= 0.5).astype(int)
        # Yspr = (np.array(Yspr) >= 0.5).astype(int)
        # Yspo = (np.array(Yspo) >= 0.5).astype(int)
        # Ps = np.array(Ps)
        # Ysp = (Ps[:,1] >= ARGS.threshold)
        Ynm = Y.copy()
        Ynm[~mask] = 0 # remove mask
        map_gt = vertexLabelsToResidueLabels(atoms, mesh, Ynm, nc=2, kdt=kdt)
        if ARGS.min_area:
            map_pr = vertexLabelsToResidueLabels(atoms, mesh, Ypo, nc=2, kdt=kdt)
        else:
            map_pr = vertexLabelsToResidueLabels(atoms, mesh, Ypr, nc=2, kdt=kdt)
        YS_gt = []
        YS_pr = []
        for key in map_gt:
            YS_gt.append(map_gt[key]["label"])
            YS_pr.append(map_pr[key]["label"])
        YS_gt = np.array(YS_gt)
        YS_pr = np.array(YS_pr)
        
    # Get selected metrics
    metrics = { "no smoothing": {}}
    if ARGS.min_area:
        metrics["smoothing"] = {}
    if not ARGS.no_residue_predictions:
        metrics["residue"] = {}
    
    for m in ARGS.metrics_eval:
        if m == "auc":
            # unsmoothed
            roc, prc = getAUC(Y, P, mask=mask)
            metrics["no smoothing"]["auroc"] = roc
            metrics["no smoothing"]["auprc"] = prc
            
            # smoothed labels
            if ARGS.min_area:
                roc, prc = getAUC(Y, P, mask=mask, mesh=mesh, threshold=ARGS.min_area)
                metrics["smoothing"]["auroc"] = roc
                metrics["smoothing"]["auprc"] = prc
            
            # # pooled labels
            # roc, prc = getAUC(Ys, Ps)
            # metrics["pooled"]["auroc"] = roc
            # metrics["pooled"]["auprc"] = prc
        elif m == 'smoothness':
            metrics["no smoothing"][m] = metric_fns[m](Ypr, E)
            if ARGS.min_area:
                metrics["smoothing"][m] = metric_fns[m](Ypo, E)
        elif m == "standard":
            # unsmoothed
            out = metric_fns[m](Y[mask], Ypr[mask])
            for o in out:
                metrics["no smoothing"][o] = out[o]
            
            # smoothed labels
            if ARGS.min_area:
                out = metric_fns[m](Y[mask], Ypo[mask])
                for o in out:
                    metrics["smoothing"][o] = out[o]
            
            # pooled labels
            # out = metric_fns[m](Ys, Ysp)
            # for o in out:
                # metrics["pooled"][o+"_P"] = out[o]
            # out = metric_fns[m](Ys, Yspr)
            # for o in out:
                # metrics["pooled"][o+"_pr"] = out[o]
            if not ARGS.no_residue_predictions:
                out = metric_fns[m](YS_gt, YS_pr)
                for o in out:
                    metrics["residue"][o] = out[o]
        else:
            # unsmoothed
            metrics["no smoothing"][m] = metric_fns[m](Y[mask], Ypr[mask])
            
            # smoothed labels
            if ARGS.min_area:
                metrics["smoothing"][m] = metric_fns[m](Y[mask], Ypo[mask])
            
            # pooled labels
            # metrics["pooled"][m+"_P"] = metric_fns[m](Ys, Ysp)
            # metrics["pooled"][m+"_pr"] = metric_fns[m](Ys, Yspr)
            if not ARGS.no_residue_predictions:
                metrics["residue"][m] = metric_fns[m](YS_gt, YS_pr)
    
    if not ARGS.no_print_individual:
        reportMetrics(metrics, header=header, label=("Protein Identifier", fname), label_width=lw, pad=6)
        printCSV(CSV, metrics, label=fname, header=header)
        header = False
    metrics_all.append(metrics)
    
    if ARGS.write_predictions:
        # save new threshold predicition
        np.savez_compressed("{}_smooth.npz".format(fname), Ygt=Y, P=P, t=ARGS.threshold, Ypr=Ypr, Ypo=Ypo, V=data['V'], F=data['F'])
    
    # save pooled predictions
    # for i in range(ids):
        # if Ys[i]:
            # print(ids[i][2], ids[i][3][1])

METRICS = {}
for m in metrics_all:
    catDicts(METRICS, m)
reduceDict(METRICS, fn=np.nanmean)
if ARGS.min_area:
    METRICS["smoothing"]["min_area"] = ARGS.min_area
    METRICS["smoothing"]["merge_distance"] = ARGS.merge_dist
reportMetrics(METRICS, header=True, label=("", "Totals"), label_width=lw, pad=6)

if ARGS.csv_file:
    printCSV(CSV, METRICS, label="means", header=True)
