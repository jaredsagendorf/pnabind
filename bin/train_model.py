#!/usr/bin/env python

# Get command-line arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("train_file",
                help="A list of training data files.")
arg_parser.add_argument("valid_file",
                help="A list of validation data files.")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="A file storing configuration options.")
arg_parser.add_argument("-p", "--write_test_predictions", action="store_true",
                help="If set will write predictions over validation set to file after training is complete.")
arg_parser.add_argument("-b", "--balance", type=str, default='balanced', choices=['balanced', 'non-masked', 'all'],
                help="Decide which set of training labels to use.")
arg_parser.add_argument("-d", "--debug", action="store_true",
                help="Print additonal stuff to the logger.")
arg_parser.add_argument("-T", "--no_tensorboard", action="store_true",
                help="Do not write any output to a tensorboard file.")
arg_parser.add_argument("-G", "--single_gpu", action="store_true",
                help="Don't distribute across multiple GPUs even if available, just use one.")
arg_parser.add_argument("-C", "--checkpoint_every", type=int, default=0,
                help="How often to write a checkpoint file to disk. Default is once at end of training.")
arg_parser.add_argument("-R", "--no_random", action="store_true",
                help="Use a fixed random seed (useful for debugging).")
arg_parser.add_argument("-W", "--no_write", action="store_true",
                help="Don't write anything to file and log everything to console.")
ARGS = arg_parser.parse_args()

# standard packages
import os
import json
import logging
import shutil
from datetime import datetime
from os.path import join as ospj

# third party modules
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataLoader, DataListLoader

# geobind modules
from geobind.nn.utils import ClassificationDatasetMemory
from geobind.nn import Trainer, Evaluator
from geobind.nn.models import NetConvEdgePool

### Load the config file
with open(ARGS.config_file) as FH:
    C = json.load(FH)

### Get run name and path
config = '.'.join(ARGS.config_file.split('.')[:-1])
run_name = "{}_{}".format(C.get("run_name", config), datetime.now().strftime("%m.%d.%Y.%H.%M"))
run_path = ospj(C.get("output_path", "."), run_name)
if(not (os.path.exists(run_path) or ARGS.no_write) ):
    os.makedirs(run_path)

### Set up logging
log_level = logging.DEBUG if ARGS.debug else logging.INFO
log_format = '%(levelname)s:    %(message)s'
if(ARGS.no_write):
    filename = None
    logging.basicConfig(format=log_format, filename=filename, level=log_level)
else:
    filename = ospj(run_path, 'run.log')
    logging.basicConfig(format=log_format, filename=filename, level=log_level)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

### Save copy of config to run directory
if(not ARGS.no_write):
    shutil.copyfile(ARGS.config_file, ospj(run_path, 'config.json'))

### Set random seed or not
if ARGS.no_random or ARGS.debug:
    np.random.seed(8)
    torch.manual_seed(0)

### Set checkpoints
if(ARGS.checkpoint_every == 0 or ARGS.checkpoint_every > C["epochs"]):
    ARGS.checkpoint_every = C['epochs'] # write once at end of training
elif(ARGS.checkpoint_every < 0):
    ARGS.checkpoint_every = False

### Load training/testing data
train_data = [_.strip() for _ in open(ARGS.train_file).readlines()]
valid_data = [_.strip() for _ in open(ARGS.valid_file).readlines()]

remove_mask = (ARGS.balance == 'all')
train_dataset = ClassificationDatasetMemory(
        train_data, C["nc"], C["data_dir"],
        balance=ARGS.balance,
        remove_mask=remove_mask,
        scale=True
    )

valid_dataset = ClassificationDatasetMemory(
        valid_data, C["nc"], C["data_dir"],
        balance='non-masked',
        remove_mask=False,
        scale=True,
        scaler=train_dataset.scaler
    )

if(torch.cuda.device_count() > 1 and (not ARGS.single_gpu)):
    # prepare data for parallelization over multiple GPUs
    DL_tr = DataListLoader(train_dataset, batch_size=C.get('batch_size', 1), shuffle=True, pin_memory=True)
    DL_vl = DataListLoader(valid_dataset, batch_size=1, shuffle=True, pin_memory=True)
else:
    # prepate data for single GPU or CPU 
    DL_tr = DataLoader(train_dataset, batch_size=C.get('batch_size', 1), shuffle=True, pin_memory=True)
    DL_vl = DataLoader(valid_dataset, batch_size=1, shuffle=True, pin_memory=True) 


### Create the model we'll be training.
nF = train_dataset.num_node_features
if C["model"]["name"] == "Net_Conv_EdgePool":
    model = NetConvEdgePool(nF, C['nc'], **C["model"]["kwargs"])

### DEBUG: model parameters
logging.debug("Model Summary:")
for name, param in model.named_parameters():
    if param.requires_grad:
        logging.debug("%s: %s", name, param.data.shape)

### set up multiple GPU utilization
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if(torch.cuda.device_count() > 1 and (not ARGS.single_gpu)):
    model = DataParallel(model)
    logging.info("Distributing model on %d gpus with root %s", torch.cuda.device_count(), device)
else:
    logging.info("Running model on device %s.", device)
model = model.to(device)
model.device = device

### set up optimizer, scheduler and loss
# optimizer
if(C["optimizer"]["name"] == "adam"):
    optimizer = torch.optim.Adam(model.parameters(), **C["optimizer"]["kwargs"])
elif(C["optimizer"]["name"] == "sgd"):
    optimizer = torch.optim.SGD(model.parameters(), **C["optimizer"]["kwargs"])
logging.info("configured optimizer: %s", C["optimizer"]["name"])

# scheduler
if(C["scheduler"]["name"] == "ReduceLROnPlateau"):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **C["scheduler"]["kwargs"])
elif(C["scheduler"]["name"] == "ExponentialLR"):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **C["scheduler"]["kwargs"])
elif(C["scheduler"]["name"] == "OneCycleLR"):
    nsteps = len(train_data)//C["batch_size"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=C["epochs"], steps_per_epoch=nsteps, **C["scheduler"]["kwargs"])
else:
    scheduler = None
logging.info("configured learning rate scheduler: %s", C["scheduler"]["name"])

# loss function
criterion = torch.nn.functional.cross_entropy

### create tensorboard writer
if(not (ARGS.no_tensorboard or ARGS.no_write) ):
    writer = SummaryWriter(run_path)
else:
    writer = None

### do the training
evaluator = Evaluator(model, post_process=torch.nn.Softmax(dim=-1))
trainer = Trainer(model, optimizer, criterion, scheduler, evaluator,
    checkpoint_path=run_path,
    writer=writer,
    quiet=False
)
trainer.train(C["epochs"], DL_tr, DL_vl, checkpoint_every=ARGS.checkpoint_every)

## write training predictions to file
#model.eval()
#if(not ARGS.no_write):
    #y_gt, prob = evaluateDataset(model, train_dataset, mask=False)
    #np.save(ospj(run_path, "train_vertex_labels.npy"), y_gt)
    #np.save(ospj(run_path, "train_vertex_probs.npy"), prob)

## evaluate test accuracy
#if(ARGS.write_test_predictions and (not ARGS.no_write)):
    #prediction_path = ospj(run_path, "predictions")
    #os.mkdir(prediction_path)

#i = 0
#use_header = True
#threshold = trainer.metrics_history['train', 'threshold'][-1]
#with torch.no_grad():
    #for test_batch in test_dataset:
        #batch, y, mask = processBatch(model, test_batch)
        #output = model(batch)
        
        ## compute predictions
        #mask = mask.cpu()
        #y = y.cpu()
        #prob = F.softmax(output, dim=1).cpu()
        #y_pr = (prob[:,1] >= threshold).long().cpu()
        
        ## compute metrics
        #metrics = getMetrics(y[mask], prob[mask], threshold=threshold)
        #report(
            #[
                #({'protein (<pdbid>.<dna_entity>_<protein_entity>.<model>)': test_data[i]}, ''),
                #(metrics, 'test metrics')
            #],
            #header=use_header
        #)
        
        ## write predictions to file
        #if(ARGS.write_test_predictions and (not ARGS.no_write)):
            #np.save(ospj(prediction_path, "%s_vertex_labels_p.npy" % (test_data[i])), y_pr)
            #np.save(ospj(prediction_path, "%s_vertex_probs.npy" % (test_data[i])), prob)
        #i += 1
        #use_header=False
