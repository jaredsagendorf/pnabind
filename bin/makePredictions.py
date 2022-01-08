import os
import sys
import argparse
import logging
import numpy as np
from sklearn.metrics import jaccard_score, balanced_accuracy_score, recall_score
from sklearn.metrics import confusion_matrix, auc, f1_score, fbeta_score
from sklearn.metrics import precision_score, matthews_corrcoef, accuracy_score
import trimesh
from glob import glob
import math

from geobind import assignMeshLabelsFromList
from geobind import vertexLabelsToResidueLabels
from geobind.nn.metrics import chooseBinaryThreshold, reportMetrics, auroc, auprc
from geobind.mesh import smoothMeshLabels
from geobind.nn.metrics import meshLabelSmoothness
from geobind.structure import getAtomKDTree, StructureData, getAtomSESA, cleanProtein
from geobind.structure.data import data as D
from geobind.mesh import Mesh

arg_parser = argparse.ArgumentParser()
# input options
arg_parser.add_argument("path",
                help="directory containing .npz files which store probabilities")
arg_parser.add_argument("--data_files",
                help="an explicit list of datafiles to use. If not provided will glob all .npz files")
arg_parser.add_argument("--model_index", type=int,
                help="use probabilities from a specific model. If not provided will average all probabilities")
arg_parser.add_argument("--pdb_dir", default=".",
                help="directory containing PDB files")
arg_parser.add_argument("--training_data",
                help="use training data to optimize threshold")

# label smoothing options
arg_parser.add_argument("--min_area", type=float,
                help="minimum cluster area")
arg_parser.add_argument("--merge_dist", type=float, default=0.0,
                help="label cluster merge distance")
arg_parser.add_argument("--no_check_intersection", action="store_true", default=False,
                help="don't check for intersection when merging label clusters")

# metrics calculation options
arg_parser.add_argument("-t", "--threshold", type=float, default=0.5,
                help="probability threshold for evaluating metrics")
arg_parser.add_argument("--metric_opt", default="ba",
                choices=["ba", "miou", "smoothness", "f1", "f2", "pr", "mcc"],
                help="metric to optimize threshold with")
arg_parser.add_argument("--metrics_eval", nargs="+", default=["miou", "ba", "auc"],
                choices=["acc", "auc", "ba", "miou", "smoothness", "f1", "f2", "standard", "sp", "fpr", "mcc", "re", "pr"],
                help="metrics to evaluate")
arg_parser.add_argument("--no_metrics", action="store_true",
                help="do not report metrics, just compute predicted labels.")
arg_parser.add_argument("--average", default='binary', choices=['binary', 'macro', 'micro', 'weighted'])

# residue prediction options
arg_parser.add_argument("--no_residue_predictions", action="store_true", 
                help="skip residue predictions")
arg_parser.add_argument("--sesa_threshold_gt", type=float,
                help="area threshold for SESA cutoffs, should be between 0 and 1")
arg_parser.add_argument("--sesa_threshold_pr", type=float,
                help="area threshold for SESA cutoffs, should be between 0 and 1")
arg_parser.add_argument("--smooth_residue_labels", action="store_true", default=False,
                help="apply smoothing to residue labels")

# output options
arg_parser.add_argument("--no_print_individual", action="store_true",
                help="don't print metrics for individual structures")
arg_parser.add_argument("--write_predictions", action="store_true", 
                help="write out predictions to file")
arg_parser.add_argument("--csv_file", 
                help="write output to CSV file")
arg_parser.add_argument("--pymol", action="store_true",
                help="write out a pymol script for residue labels")
arg_parser.add_argument("--save_residue_labels", action="store_true",
                help="write out a fasta file with residue labels")
arg_parser.add_argument("--classify_protein", action="store_true")
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
            Yp = (P[:,1] >= t[i]).astype(int)
            Yp = smoothMeshLabels(E, Yp, 2, threshold=threshold, area_faces=area_faces, faces=F)
            tn, fp, fn, tp = confusion_matrix(Y[mask], Yp[mask]).ravel()
            
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
        roc = auroc(Y[mask], P[mask], average=ARGS.average)
        prc = auprc(Y[mask], P[mask], average=ARGS.average)
    
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

def getThreshold():
    if os.path.splitext(ARGS.training_data)[1] == ".npz":
        training = np.load(ARGS.training_data)
        mask = training['Y'] >= 0
        Ytr = training['Y'][mask]
        Ptr = training['P'][mask]
        if Ptr.ndim > 2:
            Ptr = Ptr.mean(axis=-1) # reduce ensemble
    else:
        Ytr = []
        Ptr = []
        for line in open(ARGS.training_data):
            filename = os.path.join(ARGS.path, line.strip())
            data = np.load(filename)
            
            if ARGS.model_index is not None:
                P = data['Pe'][:,:,ARGS.model_index]
            else:
                P = data['Pm']
            Ytr.append(data['Y'])
            Ptr.append(P)
        Ytr = np.concatenate(Ytr)
        Ptr = np.concatenate(Ptr)
        mask = (Ytr >= 0)
        Ytr = Ytr[mask]
        Ptr = Ptr[mask]
    
    t, m = chooseBinaryThreshold(Ytr, Ptr[:,1], metric_fns[ARGS.metric_opt])
    
    if ARGS.no_metrics:
        print("Optimized threshold: {:.2f} ({:.3f} {})".format(t, m, ARGS.metric_opt))
    
    return t

def writePymol(gt, pr=None, file_name="residue_labels.pml"):
    tn = {}
    tp = {}
    fn = {}
    fp = {}
    confusion = [ [tn, fp], [fn, tp] ] # index by [y_gt][y_pr]
    colors = [ ["smudge", "red"], ["violet", "orange"]]
    for rid in gt:
        chain, resi, _ = rid.split('.')
        yi = gt[rid]["label"]
        if pr is not None:
            yj = pr[rid]["label"]
        else:
            yj = yi
        
        if chain not in confusion[yi][yj]:
            confusion[yi][yj][chain] = []
        confusion[yi][yj][chain].append(resi)
    
    FH = open(file_name, "w")
    for i in range(0, 2):
        for j in range(0, 2):
            for chain in confusion[i][j]:
                if len(confusion[i][j][chain]) > 0:
                    FH.write("color {}, (chain {} and resi {})\n".format(
                        colors[i][j],
                        chain,
                        "+".join(confusion[i][j][chain])
                    ))
    FH.close()

def writeResidueLabels(structure_id, protein, res_dict):
    FH = open("{}_labels.fasta".format(structure_id), "w")
    pdbid = structure_id.split('_')[0]
    for chain in protein.get_chains():
        res_ids = []
        res_labels = []
        seq = ""
        for res in chain:
            res_id = '{}.{}.{}'.format(chain.get_id(), res.get_id()[1], res.get_id()[2])
            if res_id in res_dict:
                label = res_dict[res_id]["label"]
            else:
                label = 0
            res_ids.append(res_id)
            res_labels.append(str(label))
            seq += D.long_to_short.get(res.get_resname(), 'X')
        FH.write(">{}_{}\n".format(pdbid, chain.get_id()))
        FH.write("{}\n".format(seq))
        FH.write("{}\n".format("".join(res_labels)))
        FH.write("{}\n".format(",".join(res_ids)))
    FH.close()

metric_fns = {
    "miou": lambda y1, y2: jaccard_score(y1, y2, average=ARGS.average, zero_division=0),
    "ba": lambda y1, y2: balanced_accuracy_score(y1, y2),
    "smoothness": lambda y, E: meshLabelSmoothness(y, E),
    "f1": lambda y1, y2: f1_score(y1, y2, average=ARGS.average),
    "f2": lambda y1, y2: fbeta_score(y1, y2, beta=2.0, average=ARGS.average),
    "standard": lambda y1, y2: confusionMatrixMetircs(y1, y2),
    "sp": lambda y1, y2: specificity(y1, y2),
    "fpr": lambda y1, y2: fpr(y1, y2),
    "pr": lambda y1, y2: precision_score(y1, y2, average=ARGS.average, zero_division=0),
    "re": lambda y1, y2: recall_score(y1, y2, average=ARGS.average, zero_division=0),
    "mcc": lambda y1, y2: float('nan') if not all([(y1 == 0).sum(), (y1 == 1).sum(), (y2 == 0).sum(), (y2 == 1).sum()]) else matthews_corrcoef(y1, y2),
    "acc": lambda y1, y2: accuracy_score(y1, y2)
}

# Decide threshold if given training data
if ARGS.training_data:
    ARGS.threshold = getThreshold()

# Set up CSV file output
if ARGS.csv_file:
    if os.path.exists(ARGS.csv_file):
        CSV_HEADER = False
        CSV = open(ARGS.csv_file, "a")
    else:
        CSV_HEADER = True
        CSV = open(ARGS.csv_file, "w")
else:
    CSV = None

# Get list of datafiles to make predictions for
if ARGS.data_files:
    file_names = [os.path.join(ARGS.path, _.strip()) for _ in open(ARGS.data_files).readlines()]
else:
    file_names = glob(os.path.join(ARGS.path, "*.npz"))

if ARGS.no_print_individual:
    lw = len("Totals")
else:
    lw = max([len(_)-len("./_predict.npz") for _ in file_names] + [len("Protein Identifier")])


# get SESA threshold
if ARGS.sesa_threshold_pr is not None:
    sesa_thresholds_pr = {}
    for res_name in D.standard_sesa_H:
    #for res_name in D.sesa_cutoffs:
        sesa_thresholds_pr[res_name] = ARGS.sesa_threshold_pr*D.standard_sesa_H[res_name]["total"]#D.sesa_cutoffs[res_name]["total"]
else:
    sesa_thresholds_pr = None

if not ARGS.no_metrics and ARGS.sesa_threshold_gt is not None:
    sesa_thresholds_gt = {}
    for res_name in D.standard_sesa_H:
    #for res_name in D.sesa_cutoffs:
        sesa_thresholds_gt[res_name] = ARGS.sesa_threshold_gt*D.standard_sesa_H[res_name]["total"]#D.sesa_cutoffs[res_name]["total"]
else:
    sesa_thresholds_gt = None

# arrays to store protein-level classification
if ARGS.classify_protein:
    YP_gt = []
    YP_en = []
    YP_sm = []

metrics_all = []
header = True
for line in file_names:
    # load data arrays
    data = np.load(line)
    V = data['V']
    F = data['F']
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if not ARGS.no_metrics:
        Y = data['Y']
        mask = (Y >= 0)
    
    if ARGS.model_index is not None:
        P = data['Pe'][:,:,ARGS.model_index]
    else:
        P = data['Pm'] if 'Pm' in data else data['P']
    
    # Get predictions for given threshold
    nc = P.shape[1]
    if nc == 2:
        Y_ensemb = (P[:,1] >= ARGS.threshold).astype(int)
    else:
        Y_ensemb = (P >= ARGS.threshold).astype(int)
        Y_ensemb = P.argmax(axis=1)
    
    if ARGS.min_area:
        # Apply smoothing
        E = mesh.edges_unique
        Y_smooth = smoothMeshLabels(E, Y_ensemb.copy(), nc, 
                threshold=ARGS.min_area,
                merge_nearby_clusters=True,
                merge_distance=ARGS.merge_dist,
                vertices=V,
                faces=F,
                area_faces=mesh.area_faces,
                check_mesh_intersection=(not ARGS.no_check_intersection),
                no_merge=[0]
        )
    
    fname = os.path.basename(line)
    fname = fname.replace("./", "")
    fname = "_".join(fname.split('_')[:-1])
    
    if not ARGS.no_residue_predictions:
        ## Get stucture based pooling
        
        # apply same processing as on training data
        pname = fname + '.pdb'
        processed = os.path.join('pdb_files', pname)
        
        if os.path.exists(processed):
            protein = StructureData(pname, path="pdb_files")
        else:
            if os.path.exists(fname + '.pdb'):
                structure = StructureData(fname+'.pdb', name=fname, path=ARGS.pdb_dir)
            else:
                structure = StructureData(fname+'.cif', name=fname, path=ARGS.pdb_dir)
            protein = getProtein(structure, D.regexes)
            
            # Clean the protein entity
            protein, pqr = cleanProtein(protein, hydrogens=True, quiet=True)
            protein.save(processed)
        
        if ARGS.min_area:
            map_pr, kdt = vertexLabelsToResidueLabels(protein, mesh, Y_smooth,
                    nc=2,
                    thresholds=sesa_thresholds_pr,
                    smooth_labels=ARGS.smooth_residue_labels,
                    ignore_class=set([1]),
                    return_kdt=True
            )
        else:
            map_pr, kdt = vertexLabelsToResidueLabels(protein, mesh, Y_ensemb,
                    nc=2,
                    thresholds=sesa_thresholds_pr,
                    smooth_labels=ARGS.smooth_residue_labels,
                    ignore_class=set([1]),
                    return_kdt=True
            )
        
        if not ARGS.no_metrics:
            # get GT labels and compute metrics
            Ynm = Y.copy()
            Ynm[~mask] = 0 # remove mask
            map_gt = vertexLabelsToResidueLabels(protein, mesh, Ynm,
                    nc=2,
                    thresholds=sesa_thresholds_gt,
                    smooth_labels=ARGS.smooth_residue_labels,
                    ignore_class=set([1]),
                    kdt=kdt
            )
            Y_struct_gt = []
            Y_struct_pr = []
            for key in map_gt:
                Y_struct_gt.append(map_gt[key]["label"])
                Y_struct_pr.append(map_pr[key]["label"])
            Y_struct_gt = np.array(Y_struct_gt)
            Y_struct_pr = np.array(Y_struct_pr)
        
        if ARGS.save_residue_labels:
            writeResidueLabels(fname, protein, map_pr)
        
        if ARGS.pymol: 
            if not ARGS.no_metrics:
                writePymol(map_gt, map_pr, file_name="{}.pml".format(fname))
            else:
                writePymol(map_pr, pr=None, file_name="{}.pml".format(fname))
    
    if not ARGS.no_metrics:
        if ARGS.classify_protein:
            YP_gt.append(np.bincount(Y[mask], minlength=nc)[1:].argmax())
            YP_en.append(np.bincount(Y_ensemb[mask], minlength=nc)[1:].argmax())
        if nc > 2:
            Y = np.eye(nc)[Y]
            Y_ensemb = np.eye(nc)[Y_ensemb]
        
        # get selected metrics
        metrics = { "no smoothing": {}}
        if ARGS.min_area:
            if ARGS.classify_protein:
                YP_sm.append(np.bincount(Y_smooth[mask], minlength=nc)[1:].argmax())
            if nc > 2:
                Y_smooth = np.eye(nc)[Y_smooth]
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
            elif m == 'smoothness':
                metrics["no smoothing"][m] = metric_fns[m](Y_ensemb, E)
                if ARGS.min_area:
                    metrics["smoothing"][m] = metric_fns[m](Y_smooth, E)
            elif m == "standard":
                # unsmoothed
                out = metric_fns[m](Y[mask], Y_ensemb[mask])
                for o in out:
                    metrics["no smoothing"][o] = out[o]
                
                # smoothed labels
                if ARGS.min_area:
                    out = metric_fns[m](Y[mask], Y_smooth[mask])
                    for o in out:
                        metrics["smoothing"][o] = out[o]
                
                if not ARGS.no_residue_predictions:
                    # residue labels
                    out = metric_fns[m](Y_struct_gt, Y_struct_pr)
                    for o in out:
                        metrics["residue"][o] = out[o]
            else:
                # unsmoothed
                metrics["no smoothing"][m] = metric_fns[m](Y[mask], Y_ensemb[mask])
                
                # smoothed labels
                if ARGS.min_area:
                    metrics["smoothing"][m] = metric_fns[m](Y[mask], Y_smooth[mask])
                
                if not ARGS.no_residue_predictions:
                    metrics["residue"][m] = metric_fns[m](Y_struct_gt, Y_struct_pr)
        
        if not ARGS.no_print_individual:
            reportMetrics(metrics, header=header, label_key="Protein Identifier", label=fname, label_width=lw, pad=6)
            printCSV(CSV, metrics, label=fname, header=header)
            header = False
        metrics_all.append(metrics)
    
    if ARGS.write_predictions:
        # save new threshold predicition (need to fix to work with no_metrics opt)
        arrays = dict(P=P, t=ARGS.threshold, Y_ensemb=Y_ensemb, V=data['V'], F=data['F'])
        if ARGS.min_area:
            arrays['Y_smooth'] = Y_smooth
        if not ARGS.no_residue_predictions:
            res_ids = filter(lambda x: x is not None, [_ if map_pr[_]["label"] == 1 else None for _ in map_pr])
            mesh = Mesh(mesh)
            Y_residue = assignMeshLabelsFromList(protein[0], mesh, res_ids,
                    weight_method='linear',
                    distance_cutoff=1.5,
                    smooth=False,
                    mask=False
            )
            arrays['Y_residue'] = Y_residue
        np.savez_compressed("{}_predict.npz".format(fname), **arrays)

if not ARGS.no_metrics:
    # write out dataset summary
    METRICS = {}
    for m in metrics_all:
        catDicts(METRICS, m)
    reduceDict(METRICS, fn=np.nanmean)
    if ARGS.min_area:
        METRICS["smoothing"]["min_area"] = ARGS.min_area
        METRICS["smoothing"]["merge_distance"] = ARGS.merge_dist
    METRICS["threshold"] = {"threshold": ARGS.threshold}
    
    if ARGS.classify_protein:
        METRICS["Protein Class"] = {"no smoothing": accuracy_score(YP_gt, YP_en)}
        if ARGS.min_area:
            METRICS["Protein Class"]["smoothing"] = accuracy_score(YP_gt, YP_sm)
    
    reportMetrics(METRICS, header=True, label_key="", label="Totals", label_width=lw, pad=6)
    
    if ARGS.csv_file:
        printCSV(CSV, METRICS, label="means", header=CSV_HEADER)
