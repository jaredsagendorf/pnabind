#!/usr/bin/env python

#### Get command-line arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("data_file",
                help="a list of data to perform inference on")
arg_parser.add_argument("data_dir",
                help="directory containing the data files")
arg_parser.add_argument("--checkpoint_files", nargs="+",
                help="a list of saved state dicts", required=True)
arg_parser.add_argument("--scalers", nargs="+", required=True,
                help="a list of pickled standard scalar objects which were fitted on training data for each model")
arg_parser.add_argument("--calibrators", nargs="+", required=False,
                help="a list of pickled calibration objects which were fitted on validation data for each model")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="file storing configuration options")
arg_parser.add_argument("--suffix", default="out",
                help="output suffix")
arg_parser.add_argument("-p", "--write_predictions", default=None,
                help="if set will write probabilities and labels to file at the given directory")
arg_parser.add_argument("-w", "--weighted_mean", action="store_true",
                help="weight models by a metric on validation data")
arg_parser.add_argument("-m", "--weight_metric", default="auprc",
                help="metric for computing weights")
arg_parser.add_argument("--no_metrics", action="store_true",
                help="disable computing any metrics")
ARGS = arg_parser.parse_args()

# standard packages
import json
import logging
import os
from os.path import join as ospj
from pickle import load
from collections import OrderedDict
from copy import deepcopy

# third party packages
import torch
import numpy as np
from torch_geometric.transforms import GenerateMeshNormals, FaceToEdge, Compose
from torch_geometric.loader import DataLoader
import sklearn.metrics as skm

# geobind packages
from geobind.nn.utils import loadDataset, loadModule
from geobind.nn import Evaluator
from geobind.nn import processBatch
from geobind.nn.metrics import auroc, auprc

METRICS = {
    "auprc": lambda y, p: auprc(y, p),
    "auroc": lambda y, p: auroc(y, p),
    "brier": lambda y, p: 1 - skm.brier_score_loss(y, p[:,1])
}

class EnsembleModel(object):
    def __init__(self, models, scalers, device, post='softmax', calibrators=None):
        self.models = models
        self.M = len(models)
        self.means = []
        self.stds = []
        self.calibrators = calibrators
        assert len(models) == len(scalers)
        if calibrators is not None:
            assert len(models) == len(calibrators)
            
        for scaler in scalers:
            self.means.append(torch.tensor(scaler.mean_, dtype=torch.float32).to(device))
            self.stds.append(torch.tensor(scaler.scale_, dtype=torch.float32).to(device))
        
        # put models in eval mode
        for model in self.models:
            model.eval()
        
        # set up post-process
        if post is None:
            # identity function
            self.post = lambda x: x
        elif post == 'softmax':
            self.post = torch.nn.Softmax(dim=-1)
    
    def forward(self, data):
        ps = []
        x = data.x
        for i in range(self.M):
            data.x = (x - self.means[i])/self.stds[i]
            output = self.models[i](data)
            output = self.post(output).cpu().numpy()
            if self.calibrators is not None:
                output = self.calibrators[i].predict_proba(output)
            ps.append(output)
        ps = np.stack(ps, axis=-1) # [V x 2 x M]
        
        return ps

def loadDataFiles(data_files, data_path, labels_key='Y', scaler=None, feature_mask=None):
    # Load the dataset passed to script
    data_files = [_.strip() for _ in open(data_files).readlines()]
    data_files = list(filter(lambda x: x[0] != '#', data_files)) # remove commented lines
    
    transforms = {
        "pre_transform": Compose([GenerateMeshNormals(), FaceToEdge(remove_faces=False)])
    }
    dataset, _, info = loadDataset(data_files, 2, labels_key, data_path,
        cache_dataset=False,
        scale=False,
        label_type="vertex",
        train_mask_keys=C.get("train_mask_keys", None),
        test_mask_keys=C.get("test_mask_keys", None),
        feature_mask=C.get("feature_mask", None),
        **transforms
    )
    DL= DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    return DL, info, data_files

def buildModel(model, state_dict, device):
    # Load checkpoint file
    checkpoint = torch.load(state_dict, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

#### Load the config file
with open(ARGS.config_file) as FH:
    C = json.load(FH)

#### Get the dataset DataLoader 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DL, info, data_files = loadDataFiles(ARGS.data_file, ARGS.data_dir, labels_key=C["labels_key"]) # get a data loader object
print("Loaded Dataset")

#### Create ensemble
# create models and load weights from file
models = []
scalers = []
calibrators = ARGS.calibrators
module, model = loadModule(C["model"]["path"], model_name=C["model"]["name"], model_args=[info["num_features"], C["nc"]], model_kwargs=C["model"]["kwargs"])

for i in range(len(ARGS.checkpoint_files)):
    # load model 
    mod = deepcopy(model)
    buildModel(mod, ARGS.checkpoint_files[i], device)
    models.append(mod)
    
    # load scaler
    scalers.append(load(open(ARGS.scalers[i], 'rb')))
    
    if calibrators:
        calibrators[i] = load(open(calibrators[i], 'rb'))

ensemble = EnsembleModel(models, scalers, device, calibrators=calibrators)
print("Created ensemble model")

if ARGS.weighted_mean and calibrators:
    weights = []
    for c in calibrators:
        weights.append(METRICS[ARGS.weight_metric](c.info['y'], c.info['p']))
    weights = np.array(weights)
    weights = weights/weights.max()
else:
    weights = np.ones(len(models))

priors = None
if ARGS.calibrators:
    priors = np.array([c.info['prior'] for c in ARGS.calibrators])

#### Loop over evaluation dataset
ROC = []
PRC = []
Ys = []
Ps = []
Ms = []
with torch.no_grad():
    i = 0
    for batch in DL:
        batch = batch.to(device)
        output = ensemble.forward(batch) # this is a [V x 2 x M] numpy array of class probalities, output[i,j,k] ~ [0,1]
        
        # get numpy arrays
        V = batch.pos.cpu().numpy()
        F = batch.face.cpu().numpy().T
        y = batch.y.cpu().numpy() if 'y' in batch else None
        mask = batch.test_mask.cpu().numpy() if 'test_mask' in batch else np.ones(len(V), dtype=bool)
        
        # combine ensemble predictions
        Pm = (output*weights).sum(axis=-1)/weights.sum()
        
        fname = data_files[i].replace("_protein_data.npz", "")
        # get metrics
        if not ARGS.no_metrics and y is not None:
            roc = auroc(y[mask], Pm[mask])
            prc = auprc(y[mask], Pm[mask])
            ROC.append(roc)
            PRC.append(prc)
            print("finished batch {} auprc: {:.3f} auroc {:.3f}".format(fname, prc, roc))
        
        if ARGS.write_predictions:
            # write prediction to file
            np.savez_compressed(ospj(ARGS.write_predictions, "{}_ensemble.npz".format(fname)), Y=y, Pm=Pm, Pe=output, V=V, F=F, mask=mask, w=weights, t=priors)
        Ys.append(y)
        Ps.append(output)
        Ms.append(mask)
        i += 1

if not ARGS.no_metrics:
    print("Vertex Label Prediction Metrics")
    print("{:8s}{:8s}{:8s}".format("metric", "mean", "std"))
    print("{:8s}{:8.3f}{:8.3f}".format("auroc", np.nanmean(ROC), np.nanstd(ROC)))
    print("{:8s}{:8.3f}{:8.3f}".format("auprc", np.nanmean(PRC), np.nanstd(PRC)))

Ys = np.concatenate(Ys, axis=0) if Ys[-1] is not None else None
Ps = np.concatenate(Ps, axis=0)
Ms = np.concatenate(Ms, axis=0)
np.savez_compressed("{}.npz".format(ARGS.suffix), Y=Ys, P=Ps, mask=Ms, w=weights, t=priors)
