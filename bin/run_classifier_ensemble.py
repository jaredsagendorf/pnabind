#!/usr/bin/env python

# standard packages
import argparse
import json
import logging
import os
from os.path import join as ospj
from pickle import load

# third party packages
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.transforms import GenerateMeshNormals
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, precision_score, recall_score

# geobind packages
from geobind.nn.utils import loadDataset, loadModule

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
arg_parser.add_argument("--suffix", default="out",
                help="Ouput suffix")
arg_parser.add_argument("-p", "--write_predictions", default=None,
                help="if set will write probabilities and labels to file at the given directory")
arg_parser.add_argument("--no_metrics", action="store_true",
                help="disable computing any metrics")
# arg_parser.add_argument("-b", "--balance", type=str, default='unmasked', choices=['balanced', 'unmasked', 'all'],
                # help="Decide which set of training labels to use.")
ARGS = arg_parser.parse_args()

class EnsembleModel(object):
    def __init__(self, models, scalers, device, post='softmax'):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.M = len(models)
        self.means = []
        self.stds = []
        
        for scaler in scalers:
            self.means.append( torch.tensor(scaler.scaler.mean_, dtype=torch.float32).to(device) )
            self.stds.append( torch.tensor(scaler.scaler.scale_, dtype=torch.float32).to(device) )
        
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
            output = self.models[i](data)[1]
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
dataset, transforms, info = loadDataset(datafiles, 2, 'Y', ARGS.data_dir,
        cache_dataset=False,
        scale=False,
        pre_transform=GMN,
        label_type="graph"
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
    _, model = loadModule(C["MODEL"]["NAME"], C["MODEL"]["PATH"], (nF,) , C["MODEL"]["KWARGS"])
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    models.append(model)
    scalers.append(load(open(ARGS.scalers[i], 'rb')))

ensemble = EnsembleModel(models, scalers, device)
print("Created ensemble model")

#### Loop over evaluation dataset
Ys = []
Ps = []
with torch.no_grad():
    i = 0
    for batch in DL:
        batch = batch.to(device)
        output = ensemble.forward(batch) # this is a [1 x 2 x M] numpy array of class probalities, output[i,j,k] ~ [0,1]
        Ys.append(batch['y'].cpu().numpy())
        Ps.append(output)
        
        # get numpy arrays
        y = batch['y'].cpu().numpy()
        V = batch['pos'].cpu().numpy()
        F = batch['face'].cpu().numpy().T
        
        # combine ensemble predictions
        Pm = output.mean(axis=-1)
        
        fname = datafiles[i].replace("_protein_data.npz", "")
        # get metrics
        #if not ARGS.no_metrics:
            #print("finished batch {} Y: {:d} Pr {:.3f}".format(fname, int(batch.y), float(Pm[0,1])))
        
        if ARGS.write_predictions:
            # write prediction to file
            np.savez_compressed(ospj(ARGS.write_predictions, "{}_ensemble.npz".format(fname)), Y=y, Pm=Pm, Pe=output, V=V, F=F)
        i += 1

Ys = np.concatenate(Ys, axis=0)
Ps = np.concatenate(Ps, axis=0).mean(axis=-1)
if not ARGS.no_metrics:
    print("{:8s}{:8s}".format("AUROC", "AUPRC"))
    print("{:8.3f}{:8.3f}".format(roc_auc_score(Ys, Ps[:,1]), average_precision_score(Ys, Ps[:,1])))
np.savez_compressed("{}.npz".format(ARGS.suffix), Y=Ys, P=Ps)
