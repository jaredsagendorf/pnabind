# third party modules
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("train_file")
arg_parser.add_argument("test_file")
arg_parser.add_argument("-c", "--config", required=True)
arg_parser.add_argument("--run_name", required=True)
ARGS = arg_parser.parse_args()

import logging
import copy
import os
import shutil
import pickle
from os.path import join as ospj
from collections import OrderedDict
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

def addWeightDecay(model, weight_decay=1e-5, skip_list=()):
    """This function excludes certain parameters (e.g. batch norm, and linear biases)
    from being subject to weight decay"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}]

def getClassWeight(dataset, nc=2):
    y = []
    for batch in dataset:
        y.append(batch.y.cpu().detach().numpy())
    #y = np.concatenate(y)
    y = np.array(y)
    weight = y.shape[0]/(nc*np.eye(nc)[y].sum(axis=0))

    return weight
    
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

def writeCSV(FH, metrics, header=True):
    items = list(sorted(metrics.items()))
    keys, vals = zip(*items)
    vals = ["{:.3f}".format(_) for _ in vals]
    if header:
        FH.write(','.join(keys) + '\n')
    FH.write(','.join(vals) + '\n')

# Params
with open(ARGS.config) as FH:
    C = json.load(FH)

# Set up ouput
run_path = ospj(C["OUTPUT_PATH"], C.get("RUN_NAME", ARGS.run_name))
if not os.path.exists(run_path):
    os.makedirs(run_path)

# Save copy of config to run directory
shutil.copyfile(ARGS.config, ospj(run_path, 'config.json'))

# Logging
filename = ospj(run_path, 'run.log')
log_format = '%(message)s'
log_level = logging.INFO
logging.basicConfig(format=log_format, filename=filename, level=log_level)
console = logging.StreamHandler()
console.setLevel(log_level)
formatter = logging.Formatter(log_format)
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Load datasets
train_datafiles = [_.strip() for _ in open(ARGS.train_file).readlines()]
valid_datafiles = [_.strip() for _ in open(ARGS.test_file).readlines()]
GMN = GenerateMeshNormals()
train_dataset, transforms, train_info = loadDataset(train_datafiles, 2, C["LABELS_KEY"], C["DATA_DIR"],
        cache_dataset=False,
        scale=True,
        pre_transform=GMN,
        label_type="graph"
)
valid_dataset, _, valid_info = loadDataset(valid_datafiles, 2, C["LABELS_KEY"], C["DATA_DIR"],
        cache_dataset=False,
        scale=True,
        label_type="graph",
        **transforms
)
class_weight = getClassWeight(train_dataset)

# save scaler to file
pickle.dump(transforms["scaler"], open(ospj(run_path, 'scaler.pkl'), "wb"))

if torch.cuda.device_count() <= 1:
    DL_tr = DataLoader(train_dataset, batch_size=C["BATCH_SIZE"], shuffle=True, pin_memory=True)
    DL_vl = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)
else:
    # prepare data for parallelization over multiple GPUs
    DL_tr = DataListLoader(train_dataset, batch_size=torch.cuda.device_count()*C["BATCH_SIZE"], shuffle=True, pin_memory=True)
    DL_vl = DataListLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Build model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
nF = train_info['num_features']
_, model = loadModule(C["MODEL"]["NAME"], C["MODEL"]["PATH"], (nF,), C["MODEL"]["KWARGS"])
class_weight = torch.tensor(class_weight, dtype=torch.float32).to(device)

# load checkpoint file if provided
if "CHECKPOINT_FILE" in C:
    prefix = "branch1."
    checkpoint = torch.load(C["CHECKPOINT_FILE"], map_location=device)
    state_dict = model.state_dict()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            name = prefix+name
            if name in checkpoint["model_state_dict"]:
                param.copy_(checkpoint["model_state_dict"][name])
model = model.to(device)

# Set up optimizer
model_parameters = addWeightDecay(model, C["WEIGHT_DECAY"])
optimizer = torch.optim.Adam(model_parameters, C["LEARNING_RATE"])
criterion = torch.nn.functional.cross_entropy

# Do training
header = True
tracked_metric = 0.75
tracked_state = None
tracked_threshold = None
TE_FILE = open(ospj(run_path, "test_metrics.csv"), "w")
TR_FILE = open(ospj(run_path, "train_metrics.csv"), "w")
for epoch in range(C["NUM_EPOCHS"]):
    # set model to training mode
    model.train()
    
    # forward + backward
    batch_loss = 0
    n = 0
    for batch in DL_tr:
        # update the model weights
        if batch.num_graphs == 1:
            continue
        batch = batch.to(device)
        
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y, weight=class_weight)
        
        # compute gradients
        loss.backward()
        optimizer.step()
        
        # add loss
        batch_loss += loss.item()
        n += 1
    batch_loss /= n
    
    if epoch % C["EVAL_EVERY"] == 0:
        if C.get("EVAL_TRAIN", True):
            mtr = evalModel(model, DL_tr)
        else:
            mtr = {}
        mtr["loss"] = batch_loss
        mvl = evalModel(model, DL_vl, threshold=mtr['t'])
        
        if mvl[C["TRACKED_METRIC"]] > tracked_metric:
            tracked_state = copy.deepcopy(model.state_dict())
            tracked_threshold = mtr["t"]
            tracked_metric = mvl[C["TRACKED_METRIC"]]
        reportMetrics({"train": mtr, "test": mvl}, label=epoch, label_key="Epoch", header=header)
        writeCSV(TE_FILE, mvl, header=header)
        writeCSV(TR_FILE, mtr, header=header)
        header = False 
TR_FILE.close()
TE_FILE.close()

# Save best state to file
fname = "best_state.tar"
if tracked_state is None:
    tracked_state = model.state_dict()
    tracked_threshold = 0.5
data = {
    'model_state_dict': tracked_state,
}
torch.save(data, ospj(run_path, fname))

# load the best state and save test/train predictions
model.load_state_dict(tracked_state)
d_tr = evalModel(model, DL_tr, threshold=tracked_threshold, return_metrics=False)
d_vl = evalModel(model, DL_vl, threshold=tracked_threshold, return_metrics=False)
np.savez_compressed(ospj(run_path, "train_predictions.npz"), **d_tr)
np.savez_compressed(ospj(run_path, "test_predictions.npz"), **d_vl)
