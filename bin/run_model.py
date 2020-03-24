#!/usr/bin/env python

# standard packages
import argparse
import json
import logging
from os.path import join as ospj
from pickle import load

# third party packages
import numpy as np

# geobind packages
from geobind.nn.utils import loadData, loadModel, evaluateDataset, getMetrics, report

#### Get command-line arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("data_file",
                help="a list of data to perform inference on")
arg_parser.add_argument("checkpoint_file",
                help="a list of data to perform inference on")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="file storing configuration options")
arg_parser.add_argument("-p", "--write_predictions", default=None,
                help="if set will write probabilities and labels to file at the given directory")
arg_parser.add_argument("-i", "--iterate_data", action="store_true",
                help="iterate over the dataset and report metrics")
arg_parser.add_argument("-s", "--summarize_data", action="store_true",
                help="summarize the dataset with a single metric")
arg_parser.add_argument("-b", "--balance", type=str, default='non-masked', choices=['balanced', 'non-masked', 'all'],
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

#### Load Scaler
scaler = load(open('scaler.pkl', 'rb'))

#### Load training/testing data
data_files = [_.strip().replace('.pdb', '_protein') for _ in open(ARGS.data_file).readlines()]

ppf_kwargs = None
if(C["model"]["kwargs"]["conv_args"]["name"] in ("Spline", "GMM", "NN", "CG")):
    ppf_kwargs = {}
    if(C["model"]["kwargs"]["conv_args"]["name"] == "Spline"):
        ppf_kwargs['scale'] = "clip"
    else:
        ppf_kwargs['scale'] = "norm"
remove_mask = ((ARGS.balance == 'all') or bool(ARGS.write_predictions))
dataset, _ = loadData(
    C["data_dir"],
    data_files, 
    scaler=scaler,
    balance=C.get("balance", ARGS.balance),
    nc=C["nc"],
    ppf_kwargs=ppf_kwargs,
    remove_mask=False,
    single_gpu=True,
    shuffle=False
)

#### Load Model
model = loadModel(C, dataset.num_features, dataset.num_classes, ARGS.checkpoint_file)

#### Evaulate Metrics
use_mask = (not bool(ARGS.write_predictions))
if(ARGS.summarize_data):
    y, prob = evaluateDataset(model, dataset, metrics=False, mask=True)
    if(ARGS.write_predictions):
        np.save(ospj(ARGS.write_predictions, "data_vertex_labels.npy"), y)
        np.save(ospj(ARGS.write_predictions, "data_vertex_probs.npy"), prob)
    metrics = getMetrics(y, prob, threshold=ARGS.threshold)
    report([(metrics, 'Entire Dataset Summary')], header=True)

if(ARGS.iterate_data or ARGS.write_predictions):
    i = 0
    use_header = True
    use_metrics = (not ARGS.no_metrics)
    for item in evaluateDataset(model, dataset, metrics=use_metrics, mask=use_mask, iterate=True, threshold=ARGS.threshold):
        if(use_metrics):
            y, prob, metrics = item
            # report metrics
            report(
                [
                    ({'protein (<pdbid>.<dna_entity>_<protein_entity>.<model>)': data_files[i]}, ''),
                    (metrics, 'per-datum metrics')
                ],
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
        
