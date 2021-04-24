#!/usr/bin/env python

# standard packages
import argparse
import json
import logging
import os
from os.path import join as ospj
from pickle import load

# third party packages
import torch
import numpy as np
from torch_geometric.transforms import Compose, FaceToEdge, PointPairFeatures, GenerateMeshNormals, Cartesian, TwoHop
from torch_geometric.data import DataLoader

# geobind packages
from geobind.nn.utils import loadDataset
from geobind.nn.models import MultiBranchNet
from geobind.nn import Evaluator
from geobind.nn import processBatch
from geobind.nn.metrics import auroc, auprc
from geobind.nn.transforms import GeometricEdgeFeatures, ScaleEdgeFeatures

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
        #super(EnsembleModel, self).__init__()
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
            output = self.models[i](data)
            output = self.post(output)
            ys.append(output.cpu().numpy())
        
        ys = np.stack(ys, axis=-1) # [V x 2 x M]
        
        return ys

def getDataTransforms(args):
    t_lookup = {
        "FaceToEdge": (FaceToEdge, lambda ob: 0),
        "GenerateMeshNormals": (GenerateMeshNormals, lambda ob: 0),
        "TwoHop": (TwoHop, lambda ob: 0),
        "PointPairFeatures": (PointPairFeatures, lambda ob: 4),
        "Cartesian": (Cartesian, lambda ob: 3),
        "GeometricEdgeFeatures": (GeometricEdgeFeatures, lambda ob: ob.edge_dim),
        "ScaleEdgeFeatures": (ScaleEdgeFeatures, lambda ob: 0)
    }
    transforms = []
    edge_dim = 0
    for arg in args:
        t = t_lookup[arg["name"]][0](**arg.get("kwargs", {}))
        edge_dim += t_lookup[arg["name"]][1](t)
        
        transforms.append(t)
    
    return Compose(transforms), edge_dim

# if ARGS.log_file is None:
    # logging.basicConfig(format='%(levelname)s:    %(message)s', level=logging.INFO)

#### Load the config file
with open(ARGS.config_file) as FH:
    C = json.load(FH)

#### Load evaluation data
datafiles = [_.strip() for _ in open(ARGS.data_file).readlines()]

trans_args = C["model"].get("transform_args", [])
transform, edge_dim = getDataTransforms(trans_args)

dataset, transforms, info = loadDataset(datafiles, C["nc"], "Y", ARGS.data_dir,
    cache_dataset=False,
    balance='unmasked',
    remove_mask=False,
    scale=False,
    pre_transform=transform
)

# wrap list of data object in a data loader
DL= DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
print("Finished loading dataset")

#### Create ensemble
# create models and load weights from file
nF = info['num_features']
models = []
scalers = []
device = torch.device("cpu") # use cpu to ensure deterministic behaviour
for i in range(len(ARGS.checkpoint_files)):
    cp = ARGS.checkpoint_files[i]
    model = MultiBranchNet(nF, C['nc'], **C["model"]["kwargs"])
    model = model.to(device)
    
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    models.append(model)
    scalers.append(load(open(ARGS.scalers[i], 'rb')))

ensemble = EnsembleModel(models, scalers, device)
print("Created ensemble model")

#### Loop over evaluation dataset
ROC = []
PRC = []
Ys = []
Ps = []
with torch.no_grad():
    i = 0
    for batch in DL:
        batch_data = processBatch(device, batch, xtras=["pos", "face"])
        batch, y, mask = batch_data['batch'], batch_data['y'], batch_data['mask']
        output = ensemble.forward(batch) # this is a [V x 2 x M] numpy array of class probalities, output[i,j,k] ~ [0,1]
        Ys.append(y)
        Ps.append(output)
        
        # get numpy arrays
        y = y.cpu().numpy()
        mask = mask.cpu().numpy()
        V = batch_data['pos'].cpu().numpy()
        F = batch_data['face'].cpu().numpy().T
        
        # combine ensemble predictions
        Pm = output.mean(axis=-1)
        
        fname = datafiles[i].replace("_protein_data.npz", "")
        # get metrics
        if not ARGS.no_metrics:
            roc = auroc(y[mask], Pm[mask])
            prc = auprc(y[mask], Pm[mask])
            ROC.append(roc)
            PRC.append(prc)
            print("finished batch {} auprc: {:.3f} auroc {:.3f}".format(fname, prc, roc))
        
        if ARGS.write_predictions:
            # write prediction to file
            np.savez_compressed(ospj(ARGS.write_predictions, "{}_ensemble.npz".format(fname)), Y=y, Pm=Pm, Pe=output, V=V, F=F, mask=mask)
        
        i += 1

if not ARGS.no_metrics:
    print("{:8s}{:8s}{:8s}".format("metric", "mean", "std"))
    print("{:8s}{:8.3f}{:8.3f}".format("auroc", np.mean(ROC), np.std(ROC)))
    print("{:8s}{:8.3f}{:8.3f}".format("auprc", np.mean(PRC), np.std(PRC)))

Ys = np.concatenate(Ys, axis=0)
Ps = np.concatenate(Ps, axis=0)
np.savez_compressed("{}.npz".format(ARGS.suffix), Y=Ys, P=Ps)
