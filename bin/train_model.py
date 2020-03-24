#!/usr/bin/env python

# standard packages
import argparse
import os
import json
import logging
import shutil
from datetime import datetime
from os.path import join as ospj
from pickle import dump

# third party packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
#from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import DataParallel

# geobind packages
from geobind.nn.models import Net_Conv_EdgePool
from geobind.nn.utils import processBatch, evaluateDataset, report, getMetrics, class_weights
from geobind.nn.utils import loadData, computePPEdgeFeatures

class History(object):
    def __init__(self):
        self._data = {}
        
    def update(self, tag, items):
        data = self._data
        if(isinstance(items, dict)):
            if(tag not in data):
                data[tag] = {}
            for key in items:
                if(key not in data[tag]):
                    data[tag][key] = []
                data[tag][key].append(items[key])
        else:
            if(tag not in data):
                data[tag] = []
            data[tag].append(items)
    
    def __getitem__(self, arg):
        tag, key = arg
        return self._data[tag][key]

class Trainer(object):
    def __init__(self, model, optimizer, criterion, checkpoint_path='.', scheduler=None, writer=None, quiet=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.writer = writer
        self.quiet = quiet
        self.checkpoint_path = checkpoint_path
        self.batch_count = 0
        self.epoch = 0
        
        # placeholder variables
        self._writer = writer
        self._first_eval = True
        
        # metrics history
        self.metrics_history = History()
    
    def train(self, nepochs, dataset, val_dataset=None, batch_loss_every=4, eval_every=2):
        # begin training
        if(not self.quiet):
            logging.info("Beginning training")
        
        for epoch in range(nepochs):
            # set model to training mode
            self.model.train()
            
            # forward + backward + update
            losses = []
            for batch in dataset:
                batch, y, mask = processBatch(self.model, batch)
                weight = class_weights(y).to(self.model.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output[mask], y[mask], weight=weight)
                loss.backward()
                self.optimizer.step()
        
                self.batch_count += 1
                losses.append(loss.item())
                
                # write batch-level stats
                if(self.batch_count % batch_loss_every  == 0):
                    if(self.writer is not None):
                        self.writer.add_scalar("train/batch_loss", loss.item(), self.batch_count)
                
                # step per-batch learning rate schedulers
                if(self.scheduler is not None):
                    if(isinstance(self.scheduler, OneCycleLR)):
                        self.scheduler.step()
            
            # adjust LR schedule on per-epoch basis
            mean_loss = sum(losses)/len(losses)
            if(self.scheduler is not None):
                if(isinstance(self.scheduler, ReduceLROnPlateau)):
                    self.scheduler.step(mean_loss)
                elif(isinstance(self.scheduler, ExponentialLR)):
                    self.scheduler.step()
            
            # metrics
            if(epoch % eval_every == 0):
                # set model to eval mode
                self.model.eval()
                metric_list = [({'epoch': self.epoch+1, 'loss': mean_loss}, "")]
                self.metrics_history.update('loss', mean_loss)
                
                # evaluate model performance
                # train set
                tr_metrics = evaluateDataset(self.model, dataset, metrics=True, choose_threshold=True)
                metric_list.append((tr_metrics, 'train'))
                if(self.writer is not None):
                    for mname in tr_metrics:
                        self.writer.add_scalar("{}/{}".format('train', mname), tr_metrics[mname], self.epoch)
                self.metrics_history.update('train', tr_metrics)
                
                # test set
                if(val_dataset is not None):
                    te_metrics = evaluateDataset(self.model, val_dataset, metrics=True, threshold=tr_metrics['threshold'])
                    metric_list.append((te_metrics, 'validation'))
                    if(self.writer is not None):
                        for mname in te_metrics:
                            self.writer.add_scalar("{}/{}".format('val', mname), te_metrics[mname], self.epoch)
                    self.metrics_history.update('validation', te_metrics)
                    
                # report performance
                report(metric_list, header=self._first_eval)
                self._first_eval = False
            
            # checkpoint
            if(ARGS.checkpoint_every and ((epoch+1) % ARGS.checkpoint_every) == 0 and (not ARGS.no_write)):
                fname = ospj(self.checkpoint_path, "{}.{}.tar".format(C['model']['name'], epoch+1))
                logging.info("Writing checkpoint to file {} at epoch {}".format(fname, epoch+1))
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': mean_loss,
                        'model_parameters': C['model']
                    }, fname)
            
            self.epoch += 1
    
    def toggle_writer(self, off=False, on=False):
        if(off):
            self.writer = None
        elif(on):
            self.writer = self._writer
        else:
            if(self.writer is None):
                self.writer = self._writer
            else:
                self.writer = None

# Get command-line arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("train_file",
                help="a list of training data")
arg_parser.add_argument("test_file",
                help="a list of test data")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="file storing configuration options")
arg_parser.add_argument("-p", "--write_test_predictions", action="store_true",
                help="if set will write final test predictions to file")
arg_parser.add_argument("-b", "--balance", type=str, default='balanced', choices=['balanced', 'non-masked', 'all'],
                help="Decide which set of training labels to use.")
arg_parser.add_argument("-T", "--no_tensorboard", action="store_true",
                help="will not write output to a tensorboard file")
#arg_parser.add_argument("-P", "--PU_postprocess", action="store_true",
                #help="perform PU learning post processing on results")
arg_parser.add_argument("-G", "--single_gpu", action="store_true",
                help="don't distribute across multiple GPUs, just use one")
arg_parser.add_argument("-C", "--checkpoint_every", type=int, default=0,
                help="How often to write a checkpoint file to disk. Default is once at end of training.")
arg_parser.add_argument("-R", "--no_random", action="store_true",
                help="Use a fixed random seed.")
arg_parser.add_argument("-W", "--no_write", action="store_true",
                help="Don't write anything to file and log everything to console.")
ARGS = arg_parser.parse_args()

# Load the config file
with open(ARGS.config_file) as FH:
    C = json.load(FH)

# Get run name and path
config = '.'.join(ARGS.config_file.split('.')[:-1])
run_name = "{}_{}".format(config, datetime.now().strftime("%m.%d.%Y.%H.%M"))
run_path = ospj(C.get("output_path", "./development"), run_name)
if(not (os.path.exists(run_path) or ARGS.no_write) ):
    os.makedirs(run_path)

# Set up logging
log_level = logging.DEBUG
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
logging.info("Configuration File:")
logging.info("\n"+json.dumps(C, indent=2))

# Save copy of config to run directory
if(not ARGS.no_write):
    shutil.copyfile(ARGS.config_file, ospj(run_path, 'config.json'))

# Set random seed or not
if(ARGS.no_random):
    np.random.seed(8)
    torch.manual_seed(0)

#### Set checkpoints
if(ARGS.checkpoint_every == 0 or ARGS.checkpoint_every > C["epochs"]):
    ARGS.checkpoint_every = C['epochs'] # write once at end of training
elif(ARGS.checkpoint_every < 0):
    ARGS.checkpoint_every = False

#### Load training/testing data
train_data = [_.rsplit('.', 1)[0]+"_protein" for _ in open(ARGS.train_file).readlines()]
test_data = [_.rsplit('.', 1)[0]+"_protein" for _ in open(ARGS.test_file).readlines()]

ppf_kwargs = None
if(C["model"]["kwargs"]["conv_args"]["name"] in ("Spline", "GMM", "NN", "CG")):
    ppf_kwargs = {}
    if(C["model"]["kwargs"]["conv_args"]["name"] == "Spline"):
        ppf_kwargs['scale'] = "clip"
    else:
        ppf_kwargs['scale'] = "norm"
remove_mask = (ARGS.balance == 'all')
train_dataset, scaler = loadData(C["data_dir"], train_data, 
    batchsize=C.get("batch_size", 1),
    balance=C.get("balance", ARGS.balance),
    nc=C["nc"],
    ppf_kwargs=ppf_kwargs,
    single_gpu=ARGS.single_gpu,
    remove_mask=remove_mask
)
test_dataset, _ = loadData(C["data_dir"], test_data,
    scaler=scaler,
    balance='non-masked',
    batchsize=1,
    shuffle=False,
    nc=C["nc"],
    ppf_kwargs=ppf_kwargs,
    single_gpu=ARGS.single_gpu
)

#### Save scaler to file
if(not ARGS.no_write):
    dump(scaler, open(ospj(run_path, "scaler.pkl"), "wb"))

##### Create model
nF = train_dataset.num_features
nC = train_dataset.num_classes
if(C["model"]["name"] == "Net_Conv_EdgePool"):
    model = Net_Conv_EdgePool(nF, nC, **C["model"]["kwargs"])

# DEBUG: model parameters
logging.debug("Model Summary:")
for name, param in model.named_parameters():
    if param.requires_grad:
        logging.debug("%s: %s", name, param.data.shape)

#### set up multiple GPU utilization
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if(torch.cuda.device_count() > 1 and (not ARGS.single_gpu)):
    model = DataParallel(model)
    logging.info("Distributing model on %d gpus with root %s", torch.cuda.device_count(), device)
else:
    logging.info("Running model on device %s.", device)
model = model.to(device)
model.device = device

#### set up optimizer scheduler and loss
# optimizer
if(C["optimizer"]["name"] == "adam"):
    optimizer = torch.optim.Adam(model.parameters(), **C["optimizer"]["kwargs"])
elif(C["optimizer"]["name"] == "sgd"):
    optimizer = torch.optim.SGD(model.parameters(), **C["optimizer"]["kwargs"])
logging.info("configured optimizer: %s", C["optimizer"]["name"])

# scheduler
if(C["scheduler"]["name"] == "ReduceLROnPlateau"):
    scheduler = ReduceLROnPlateau(optimizer, **C["scheduler"]["kwargs"])
elif(C["scheduler"]["name"] == "ExponentialLR"):
    scheduler = ExponentialLR(optimizer, **C["scheduler"]["kwargs"])
elif(C["scheduler"]["name"] == "OneCycleLR"):
    nsteps = len(train_data)//C["batch_size"]
    scheduler = OneCycleLR(optimizer, epochs=C["epochs"], steps_per_epoch=nsteps, **C["scheduler"]["kwargs"])
else:
    scheduler = None
logging.info("configured learning rate scheduler: %s", C["scheduler"]["name"])

# loss function
criterion = F.cross_entropy

#### create tensorboard writer
if(not (ARGS.no_tensorboard or ARGS.no_write) ):
    writer = SummaryWriter(run_path)
else:
    writer = None

#### do the training
trainer = Trainer(model, optimizer, criterion,
    checkpoint_path=run_path,
    scheduler=scheduler,
    writer=writer,
    quiet=False
)
trainer.train(C["epochs"], train_dataset, val_dataset=test_dataset)

# write training predictions to file
model.eval()
if(not ARGS.no_write):
    y_gt, prob = evaluateDataset(model, train_dataset, mask=False)
    np.save(ospj(run_path, "train_vertex_labels.npy"), y_gt)
    np.save(ospj(run_path, "train_vertex_probs.npy"), prob)

# evaluate test accuracy
if(ARGS.write_test_predictions and (not ARGS.no_write)):
    prediction_path = ospj(run_path, "predictions")
    os.mkdir(prediction_path)

i = 0
use_header = True
threshold = trainer.metrics_history['train', 'threshold'][-1]
with torch.no_grad():
    for test_batch in test_dataset:
        batch, y, mask = processBatch(model, test_batch)
        output = model(batch)
        
        # compute predictions
        mask = mask.cpu()
        y = y.cpu()
        prob = F.softmax(output, dim=1).cpu()
        y_pr = (prob[:,1] >= threshold).long().cpu()
        
        # compute metrics
        metrics = getMetrics(y[mask], prob[mask], threshold=threshold)
        report(
            [
                ({'protein (<pdbid>.<dna_entity>_<protein_entity>.<model>)': test_data[i]}, ''),
                (metrics, 'test metrics')
            ],
            header=use_header
        )
        
        # write predictions to file
        if(ARGS.write_test_predictions and (not ARGS.no_write)):
            np.save(ospj(prediction_path, "%s_vertex_labels_p.npy" % (test_data[i])), y_pr)
            np.save(ospj(prediction_path, "%s_vertex_probs.npy" % (test_data[i])), prob)
        i += 1
        use_header=False
