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
from torch_geometric.transforms import GenerateMeshNormals
from torch_geometric.data import DataLoader, DataListLoader, Data

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import average_precision_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, jaccard_score
from sklearn.preprocessing import StandardScaler

# geobind modules
from geobind.nn.utils import balancedClassIndices, addWeightDecay, loadModule
from geobind.nn.metrics import reportMetrics, chooseBinaryThreshold

class NodeScaler(object):
    def __init__(self):
        self._data_arrays = []
        self.scaler = StandardScaler()
    
    def update(self, array):
        self._data_arrays.append(array)
    
    def fit(self):
        self.scaler.fit(np.concatenate(self._data_arrays, axis=0))
    
    def scale(self, array):
        return self.scaler.transform(array)

def loadDataset(data_files, ncs, label_keys, data_dir ,
        balance="unmasked",
        remove_mask=False,
        unmasked_class=0,
        scaler=None,
        scale=True,
        transform=None,
        pre_filter=None,
        pre_transform=None,
        feature_mask=None,
        percentage=1.0
    ):
    
    if isinstance(data_files, str):
        with open(data_files) as FH:
            data_files = [_.strip() for _ in FH.readlines()]
    data_files = [ospj(data_dir, f) for f in data_files]
    data_list = []
    
    if isinstance(label_keys, str):
        # convert to list
        labels_keys = [labels_keys]
    
    # read and process datafiles
    for f in data_files:
        data_arrays = np.load(f)
        
        Y = {}
        for i in range(len(label_keys)):
            lk = label_keys[i]
            Y[lk] = data_arrays[lk]
            if len(Y[lk].shape) > 0:
                if remove_mask:
                    # remove any previous masking
                    Y[(Y == -1)] = unmasked_class
                
                if balance == 'balanced':
                    idxb = balancedClassIndices(data_arrays[lk], range(ncs[i]), max_percentage=percentage)
                elif balance == 'unmasked':
                    idxb = (data_arrays[lk] >= 0)
                elif balance == 'all':
                    idxb = (data_arrays[lk] == data_arrays[lk])
                else:
                    raise ValueError("Unrecognized value for `balance` keyword: {}".format(balance))
            Y[lk] = torch.tensor(Y[lk], dtype=torch.int64)
        
        if feature_mask is not None:
            X = data_arrays['X'][:, feature_mask]
        else:
            X = data_arrays['X']
        
        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            pos=torch.tensor(data_arrays['V'], dtype=torch.float32),
            norm=torch.tensor(data_arrays['N'], dtype=torch.float32),
            face=torch.tensor(data_arrays['F'].T, dtype=torch.int64),
            edge_attr=None,
            edge_index=None,
            **Y
        )
        data.mask = torch.tensor(idxb, dtype=torch.bool)
        data_list.append(data)
    
    # filter data
    if pre_filter is not None:
        data_list = [data for data in data_list if pre_filter(data)]
    
    # transform data
    if pre_transform is not None:
        data_list = [pre_transform(data) for data in data_list]
    
    # scale data
    if scale:
        # build a scaler
        if scaler is None:
            scaler = NodeScaler()
            for data in data_list:
                scaler.update(data.x[data.mask])
            scaler.fit()
        
        # scale node features in each data object
        for data in data_list:
            data.x = torch.tensor(scaler.scale(data.x), dtype=torch.float32)
    
    transforms = {
        "scaler": scaler,
        "transform": transform,
        "pre_transform": pre_transform,
        "pre_filter": pre_filter
    }
    
    info = {
        "num_features": int(data_list[0].x.shape[1]),
        "num_classes": ncs,
        "num_instances": len(data_list)
    }
    return data_list, transforms, info

def getClassWeight(dataset, nc=2, y_key='y', use_mask=True):
    y = []
    for batch in dataset:
        y.append(batch[y_key].cpu().detach().numpy().reshape(-1,1))
    y = np.concatenate(y)
    if use_mask:
        y = y[y >= 0]
    
    weight = y.shape[0]/(nc*np.eye(nc)[y].sum(axis=0))
    
    return weight

def evalModel(model, dataset, threshold=None, return_metrics=True, y_key='y', bi=0, nc=2, use_mask=True):
    model.eval()
    
    y_gt = []
    P = []
    
    for batch in dataset:
        batch = batch.to(device)
        output = model(batch)
        output = torch.nn.functional.softmax(output[bi], dim=1)
        
        y_gt.append(batch[y_key].cpu().detach().numpy())
        P.append(output.cpu().detach().numpy())
    
    y_gt = np.concatenate(y_gt)
    P = np.concatenate(P)
    
    if use_mask:
        mask = (y_gt >= 0)
        y_gt = y_gt[mask]
        P = P[mask]
    
    if nc == 2:
        if threshold is None:
            threshold, _ = chooseBinaryThreshold(y_gt, P[:,1], balanced_accuracy_score)
        y_pr = (P[:,1] >= threshold)
    else:
        if threshold is None:
            threshold = 0.333
        y_pr = P.argmax(axis=1)
    
    if return_metrics:
        if nc == 2:
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
        else:
            Y_gt = np.eye(nc)[y_gt]
            Y_pr = np.eye(nc)[y_pr]
            metrics = {
                "ROC": roc_auc_score(Y_gt, P, average="macro", multi_class="ovo"),
                "APRC": average_precision_score(Y_gt, P, average="macro"),
                "PR": recall_score(Y_gt, Y_pr, average="macro", zero_division=0),
                "RE": precision_score(Y_gt, Y_pr, average="macro", zero_division=0),
                "MIOU": jaccard_score(Y_gt, Y_pr, average="macro", zero_division=0),
                "AC": accuracy_score(Y_gt, Y_pr),
                "MCC": matthews_corrcoef(y_gt, y_pr),
                "BA": balanced_accuracy_score(y_gt, y_pr),
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

def setup_logger(name, log_file, formatter, level=logging.INFO):
    """To setup as many loggers as you want"""
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

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
filename1 = ospj(run_path, 'run.log1')
filename2 = ospj(run_path, 'run.log2')
formatter = logging.Formatter('%(message)s')
log_level = logging.INFO

logger1 = setup_logger('log1', filename1, formatter, log_level)
logger2 = setup_logger('log2', filename2, formatter, log_level)

# add console
console = logging.StreamHandler()
console.setLevel(log_level)
console.setFormatter(formatter)
logger2.addHandler(console)

# Load datasets
train_datafiles = [_.strip() for _ in open(ARGS.train_file).readlines()]
valid_datafiles = [_.strip() for _ in open(ARGS.test_file).readlines()]
GMN = GenerateMeshNormals()
train_dataset, transforms, train_info = loadDataset(train_datafiles, [C["nc1"], C["nc2"]], ["y1", "y2"], C["DATA_DIR"],
        balance="balanced",
        scale=True,
        pre_transform=GMN,
)
valid_dataset, _, valid_info = loadDataset(valid_datafiles, [C["nc1"], C["nc2"]], ["y1", "y2"], C["DATA_DIR"],
        balance="balanced",
        scale=True,
        **transforms
)

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

if not C.get("MASK_BATCHWISE", False):
    class_weight1 = getClassWeight(train_dataset, nc=C["nc1"], y_key="y1")
    class_weight1 = torch.tensor(class_weight1, dtype=torch.float32).to(device)
else:
    class_weight1 = None
class_weight2 = getClassWeight(train_dataset, nc=C["nc2"], y_key="y2")
class_weight2 = torch.tensor(class_weight2, dtype=torch.float32).to(device)
#logging.getLogger("log2").info("%s", class_weight1)
#logging.getLogger("log2").info("%s", class_weight2)

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
                print(name)
model = model.to(device)

# Set up optimizer
model_parameters = addWeightDecay(model, C["WEIGHT_DECAY"])
optimizer = torch.optim.Adam(model_parameters, C["LEARNING_RATE"])
criterion = torch.nn.functional.cross_entropy
#wl1 = torch.tensor(1-C["alpha"], dtype=torch.float32, requires_grad=False)
#wl2 = torch.tensor(C["alpha"], dtype=torch.float32, requires_grad=False)
wn = lambda n: n*(1-C["alpha"])/C["NUM_EPOCHS"] + C["alpha"] 

# Do training
header = True
tracked_metric = 0.75
tracked_state = None
tracked_threshold = None
TE_FILE1 = open(ospj(run_path, "test_metrics1.csv"), "w")
TR_FILE1 = open(ospj(run_path, "train_metrics1.csv"), "w")
TE_FILE2 = open(ospj(run_path, "test_metrics2.csv"), "w")
TR_FILE2 = open(ospj(run_path, "train_metrics2.csv"), "w")
for epoch in range(C["NUM_EPOCHS"]):
    # set model to training mode
    model.train()
    
    # set loss weight
    alpha = torch.tensor(wn(epoch), dtype=torch.float32, requires_grad=False)
    
    # forward + backward
    batch_loss = []
    batch_loss1 = []
    batch_loss2 = []
    n = 0
    for batch in DL_tr:
        # update the model weights
        if batch.num_graphs == 1:
            continue
        batch = batch.to(device)
        
        optimizer.zero_grad()
        output = model(batch)
        if C.get("MASK_BATCHWISE", False):
            idxb = balancedClassIndices(batch['y1'].cpu().numpy(), range(C["nc1"]))
            mask = torch.tensor(idxb, dtype=torch.bool)
        else:
            mask = batch.mask
        loss1 = criterion(output[0][mask], batch['y1'][mask], weight=class_weight1)
        loss2 = criterion(output[1], batch['y2'], weight=class_weight2)
        if torch.isnan(loss1):
            # entire batch was masked
            loss = loss2
        else:
            loss = (1-alpha)*loss1 + alpha*loss2
        
        # compute gradients
        loss.backward()
        optimizer.step()
        
        # add loss
        batch_loss.append(loss.item())
        batch_loss1.append(loss1.item())
        batch_loss2.append(loss2.item())
        n += 1
    batch_loss = np.nanmean(batch_loss)
    batch_loss1 = np.nanmean(batch_loss1)
    batch_loss2 = np.nanmean(batch_loss2)
    
    if epoch % C["EVAL_EVERY"] == 0:
        if C.get("EVAL_TRAIN", True):
            mtr1 = evalModel(model, DL_tr, y_key='y1', bi=0, nc=C["nc1"])
            mtr2 = evalModel(model, DL_tr, y_key='y2', bi=1, nc=C["nc2"])
        else:
            mtr1 = {}
            mtr2 = {}
        mtr1["loss"] = batch_loss1
        mtr2["loss"] = batch_loss2
        
        mvl1 = evalModel(model, DL_vl, y_key='y1', bi=0, nc=C["nc1"])
        mvl2 = evalModel(model, DL_vl, threshold=mtr2['t'], y_key='y2', bi=1, nc=C["nc2"])
        
        if mvl2[C["TRACKED_METRIC"]] > tracked_metric:
            tracked_state = copy.deepcopy(model.state_dict())
            tracked_threshold = mtr2["t"]
            tracked_metric = mvl2[C["TRACKED_METRIC"]]
        reportMetrics({"train": mtr1, "test": mvl1}, label=epoch, label_key="Epoch", header=header, logger="log1")
        reportMetrics({"train": mtr2, "test": mvl2}, label=epoch, label_key="Epoch", header=header, logger="log2")
        writeCSV(TE_FILE1, mvl1, header=header)
        writeCSV(TR_FILE1, mtr1, header=header)
        writeCSV(TE_FILE2, mvl2, header=header)
        writeCSV(TR_FILE2, mtr2, header=header)
        header = False 
TR_FILE1.close()
TE_FILE1.close()
TR_FILE2.close()
TE_FILE2.close()

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

d_tr1 = evalModel(model, DL_tr, threshold=tracked_threshold, return_metrics=False, y_key='y1', bi=0, nc=C["nc1"])
d_vl1 = evalModel(model, DL_vl, threshold=tracked_threshold, return_metrics=False, y_key='y1', bi=0, nc=C["nc1"])
np.savez_compressed(ospj(run_path, "train_predictions1.npz"), **d_tr1)
np.savez_compressed(ospj(run_path, "test_predictions1.npz"), **d_vl1)

d_tr2 = evalModel(model, DL_tr, threshold=tracked_threshold, return_metrics=False, y_key='y2', bi=1, nc=C["nc2"])
d_vl2 = evalModel(model, DL_vl, threshold=tracked_threshold, return_metrics=False, y_key='y2', bi=1, nc=C["nc2"])
np.savez_compressed(ospj(run_path, "train_predictions2.npz"), **d_tr2)
np.savez_compressed(ospj(run_path, "test_predictions2.npz"), **d_vl2)
