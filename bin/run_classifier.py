import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("test_file")
arg_parser.add_argument("scaler_file")
arg_parser.add_argument("-c", "--config", required=True)
ARGS = arg_parser.parse_args()

import logging
import copy
import os
import shutil
import pickle
from os.path import join as ospj
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import GenerateMeshNormals
from torch_geometric.data import DataLoader, DataListLoader

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_recall_curve, auc

# geobind modules
from geobind.nn.utils import loadDataset, loadModule
from geobind.nn.metrics import reportMetrics, chooseBinaryThreshold

def evalModel(model, dataset, threshold=None, return_metrics=True):
    model.eval()
    y_gt = []
    P = []
    
    for batch in dataset:
        batch = batch.to(device)
        output = F.softmax(model(batch), dim=1)
        
        y_gt.append(batch.y.cpu().detach().numpy())
        P.append(output.cpu().detach().numpy())
    
    y_gt = np.concatenate(y_gt)
    P = np.concatenate(P)
    
    if threshold is None:
        threshold, _ = chooseBinaryThreshold(y_gt, P[:,1], balanced_accuracy_score)
    y_pr = (P[:,1] >= threshold)
    
    if return_metrics:
        pr, re, t = precision_recall_curve(y_gt, P[:,1])
        metrics = {
            "ROC": roc_auc_score(y_gt, P[:,1]),
            "PRC": auc(re, pr),
            "BA": balanced_accuracy_score(y_gt, y_pr),
            "AC": accuracy_score(y_gt, y_pr),
            "PR": recall_score(y_gt, y_pr),
            "RE": precision_score(y_gt, y_pr, zero_division=0),
            "MCC": matthews_corrcoef(y_gt, y_pr),
            "t": threshold
        }
        
        return metrics
    else:
        return {"y_gt": y_gt, "y_pr": y_pr, "p": P, "t": threshold}

# Params
with open(ARGS.config) as FH:
    C = json.load(FH)

# Logging
log_format = '%(message)s'
log_level = logging.INFO
logging.basicConfig(format=log_format, level=log_level)

# Load datasets
datafiles = [_.strip() for _ in open(ARGS.test_file).readlines()]
GMN = GenerateMeshNormals()
scaler = pickle.load(open(ARGS.scaler_file, 'rb'))
transforms = {
    "scaler": scaler,
    "pre_transform": GMN
}
dataset, _, info = loadDataset(datafiles, 2, C["LABELS_KEY"], C["DATA_DIR"],
    cache_dataset=False,
    scale=True,
    label_type="graph",
    **transforms
)
DL= DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

# Build model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
nF = info['num_features']
_, model = loadModule(C["MODEL"]["NAME"], C["MODEL"]["PATH"], (nF,), C["MODEL"]["KWARGS"])

# Load checkpoint file if provided
if "CHECKPOINT_FILE" in C:
    checkpoint = torch.load(C["CHECKPOINT_FILE"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

# Evaluate test dataset
with torch.no_grad():
    metrics = evalModel(model, DL, threshold=0.5)    
    reportMetrics({"test": metrics}, header=True)
