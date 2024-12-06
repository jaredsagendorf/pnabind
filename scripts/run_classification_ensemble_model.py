#!/usr/bin/env python

# standard packages
import argparse

#### Get command-line arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("data_file",
                help="a list of data to perform inference on")
arg_parser.add_argument("data_dir",
                help="directory containing the data files")
arg_parser.add_argument("--checkpoint_files", nargs="+",
                help="a list of saved state dicts", required=True)
arg_parser.add_argument("--scalers", nargs="+", required=True,
                help="a list of pickled standard scalar objects which were fitted on training data for each model")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="file storing configuration options")
arg_parser.add_argument("--threshold", default=0.5, type=float,
                help="probability threshold")
arg_parser.add_argument("--prediction", action="store_true",
                help="this will ignore any ground-truth labels and only output predicted labels.")
ARGS = arg_parser.parse_args()

import json
import os
from os.path import join as ospj
from pickle import load
from copy import copy

# third party packages
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.transforms import GenerateMeshNormals
from torch_geometric.data import DataLoader

# pnabind packages
from pnabind.nn.utils import loadDataset, loadModule

class EnsembleModel(object):
    def __init__(self, models, scalers, device, post='softmax'):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.M = len(models)
        self.means = []
        self.stds = []
        
        for scaler in scalers:
            self.means.append( torch.tensor(scaler.mean_, dtype=torch.float32).to(device) )
            self.stds.append( torch.tensor(scaler.scale_, dtype=torch.float32).to(device) )
        
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
        ys = []
        x = data.x
        for i in range(self.M):
            data.x = (x - self.means[i])/self.stds[i]
            output = self.models[i](data.x, data.pos, data.norm, data.batch)
            output = self.post(output)
            ys.append(output.cpu().numpy())
        
        ys = np.stack(ys, axis=-1) # [V x 2 x M]
        
        return ys

#### Load the config file
with open(ARGS.config_file) as FH:
    C = json.load(FH)

#### Load evaluation data
datafiles = [_.strip() for _ in open(ARGS.data_file).readlines()]
datafiles = list(filter(lambda x: x[0] != '#', datafiles)) # remove commented lines

GMN = GenerateMeshNormals()
if ARGS.prediction
    y_key = None
else:
    y_key = C.get("LABELS_KEY", None)
dataset, transforms, info = loadDataset(datafiles, 2, y_key, ARGS.data_dir,
        cache_dataset=False,
        scale=False,
        pre_transform=GMN,
        label_type="graph",
        feature_mask=C.get("FEATURE_MASK",None)
)

# wrap list of data object in a data loader
DL= DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
print("Finished loading dataset")

#### Create ensemble
# create models and load weights from file
nF = info['num_features']
models = []
scalers = []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # use cpu to ensure deterministic behaviour
for i in range(len(ARGS.checkpoint_files)):
    cp = ARGS.checkpoint_files[i]
    _, model = loadModule(C["MODEL"]["PATH"], model_name=C["MODEL"]["NAME"], model_args=(nF,), model_kwargs=C["MODEL"]["KWARGS"])
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    models.append(model)
    scalers.append(load(open(ARGS.scalers[i], 'rb')))

ensemble = EnsembleModel(models, scalers, device)
print("Created ensemble model")

#### Loop over evaluation dataset
OUT = open("predictions.csv", "w")
OUT.write("sequence_id,Y_gt,Y_pr,P\n")
Ys = []
Ps = []
with torch.no_grad():
    for idx, batch in enumerate(DL):
        batch = batch.to(device)
        output = ensemble.forward(batch) # this is a [1 x 2 x M] numpy array of class probalities, output[i,j,k] ~ [0,1]
        
        # get numpy arrays
        if 'y' in batch:
            y = batch['y'].cpu().numpy()
        else:
            y = -1
        Ys.append([y])
        Ps.append(output)
        
        # combine ensemble predictions
        Pm = output.mean(axis=-1)[0]
        
        # write prediction to file
        fname = datafiles[idx].replace("_protein_data.npz", "")
        OUT.write("{},{},{},{:.3f}\n".format(fname, int(y), int(Pm[1] > ARGS.threshold), Pm[1]))
OUT.close()
Ys = np.concatenate(Ys, axis=0)
Ps = np.concatenate(Ps, axis=0)
np.savez_compressed("predictions.npz", Y=Ys, P=Ps)
