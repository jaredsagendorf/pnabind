#!/usr/bin/env python

# standard packages
import argparse
import json
import logging
from os.path import join as ospj
from pickle import load

# third party packages
import torch
import numpy as np
from torch_geometric.transforms import Compose, FaceToEdge
from torch_geometric.data import DataLoader

# geobind packages
from geobind.nn.utils import loadDataset
from geobind.nn.models import NetConvPool, PointNetPP, MultiBranchNet
from geobind.nn import Evaluator
from geobind.nn import processBatch
from geobind.nn.transforms import GeometricEdgeFeatures, ScaleEdgeFeatures
from geobind.nn.metrics import reportMetrics

#### Get command-line arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("data_file",
                help="a list of data to perform inference on")
arg_parser.add_argument("checkpoint_file",
                help="saved model parameters")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="file storing configuration options")
arg_parser.add_argument("-p", "--write_predictions", default=None,
                help="if set will write probabilities and labels to file at the given directory")
arg_parser.add_argument("-i", "--iterate_data", action="store_true",
                help="iterate over the dataset and report metrics")
arg_parser.add_argument("-s", "--summarize_data", action="store_true",
                help="summarize the dataset with a single metric")
arg_parser.add_argument("-b", "--balance", type=str, default='unmasked', choices=['balanced', 'unmasked', 'all'],
                help="Decide which set of training labels to use.")
arg_parser.add_argument("-l", "--log_file", default=None,
                help="write log info to given file.")
arg_parser.add_argument("-t", "--threshold", type=float, default=0.5,
                help="threshold for evaluating metrics.")
arg_parser.add_argument("-M", "--no_metrics", action="store_true",
                help="Don't evaluate metrics. Only used when --write_predicitons is set.")
ARGS = arg_parser.parse_args()

if(ARGS.log_file is None):
    logging.basicConfig(format='%(levelname)s:    %(message)s', level=logging.INFO)

#### Load the config file
with open(ARGS.config_file) as FH:
    C = json.load(FH)

#### Load training/testing data
datafiles = [_.strip() for _ in open(ARGS.data_file).readlines()]

remove_mask = (C.get("balance", ARGS.balance) == 'all')
dataset, transforms, info = loadDataset(datafiles, C["nc"], C["labels_key"], C["data_dir"],
        cache_dataset=C.get("cache_dataset", False),
        balance=C.get("balance", ARGS.balance),
        remove_mask=remove_mask,
        scale=True,
        pre_transform=Compose([FaceToEdge(remove_faces=False), GeometricEdgeFeatures(), ScaleEdgeFeatures(method=C["model"]["kwargs"].get("scale_edge_features", None))])
    )

# prepate data for CPU
DL= DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

#### Load Model
# Create the model we'll be training.
nF = info['num_features']
if C["model"]["name"] == "NetConvPool":
    model = NetConvPool(nF, C['nc'], **C["model"]["kwargs"])
elif C["model"]["name"] == "PointNetPP":
    model = PointNetPP(nF, C['nc'], **C["model"]["kwargs"])
elif C["model"]["name"] == "MultiBranchNet":
    model = MultiBranchNet(nF, C['nc'], **C["model"]["kwargs"])

device = torch.device('cpu')
model = model.to(device)

checkpoint = torch.load(ARGS.checkpoint_file, map_location=device)
###
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint["model_state_dict"].items():
    name = k.replace("module.", "") # remove module
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
###
#model.load_state_dict(checkpoint["model_state_dict"])

#i = 0
#for batch in DL:
    ##if datafiles[i] == "4jrq.C1_A(A).0_protein_data.npz":
    ##if datafiles[i] == "1zzi.C1_A(A).0_protein_data.npz":
    ##if datafiles[i] == "5zkl.B1_A(A).0_protein_data.npz":
    #print("+++++++++++++++++++")
    #print(datafiles[i])
    #batch_data = processBatch(device, batch)
    #batch = batch_data['batch']
    #model.forward(batch, save_pools=True, prefix=datafiles[i].replace("_protein_data.npz", ""))
    ##model.forward(batch, save_pools=True, prefix="batch")
    #i += 1

#exit(0)
#### Evaulate Metrics
evaluator = Evaluator(model, C["nc"], device=device, post_process=torch.nn.Softmax(dim=-1))
use_mask = (not bool(ARGS.write_predictions))
if ARGS.summarize_data:
    out = evaluator.eval(DL, use_masks=False, return_masks=True)
    y = out['y']
    prob = out['output']
    masks = out['masks']
    if ARGS.write_predictions:
        np.save(ospj(ARGS.write_predictions, "data_vertex_labels.npy"), y)
        np.save(ospj(ARGS.write_predictions, "data_vertex_probs.npy"), prob)
    metrics = evaluator.getMetrics(y, prob, masks, threshold=ARGS.threshold)
    reportMetrics({'Entire Dataset Summary': metrics} , header=True)

if ARGS.iterate_data or ARGS.write_predictions:
    i = 0
    use_header = True
    use_metrics = (not ARGS.no_metrics)
    for item in evaluateDataset(model, dataset, metrics=use_metrics, mask=use_mask, iterate=True, threshold=ARGS.threshold):
        if(use_metrics):
            y, prob, metrics = item
            # report metrics
            reportMetrics(
                {'per-datum metrics': metrics},
                label=('protein (<pdbid>.<dna_entity>_<protein_entity>.<model>)', data_files[i]),
                header=use_header
            )
        else:
            y, prob = item
        
        # write predictions to file
        if(ARGS.write_predictions):
            # compute predictions
            y_pr = (prob[:,1] >= ARGS.threshold).long().cpu()
            np.save(ospj(ARGS.write_predictions, "%s_vertex_labels_p.npy" % (data_files[i])), y_pr)
            np.save(ospj(ARGS.write_predictions, "%s_vertex_probs.npy" % (data_files[i])), prob)
        i += 1
        use_header = False
