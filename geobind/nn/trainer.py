# builtin modules
import logging
from os.path import join as ospj

# third part modules
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, ExponentialLR
from torch_geometric.nn import DataParallel

# geobind modules
from geobind.nn.utils import classWeights
from geobind.nn import processBatch
from geobind.nn.metrics import reportMetrics

class Scheduler(object):
    def __init__(self, scheduler):
        self.epoch = 0
        self.scheduler = scheduler
        self.history = {
            "loss": 0,
            "batch_count": 0
        }
    
    def step(self, epoch, loss, **kwargs):
        # per-epoch schedulers
        new_epoch = (epoch > self.epoch)
        if new_epoch:
            # we are in a new epoch, update per-epoch schedulers
            self.epoch = epoch
            if isinstance(self.scheduler, ReduceLROnPlateau):
                mean_loss = self.history['loss']/self.history["batch_count"]
                self.scheduler.step(mean_loss)
                self.history["loss"] = 0
                self.history["batch_count"] = 0
            elif isinstance(self.scheduler, ExponentialLR):
                self.scheduler.step()
        
        # per-batch schedulers
        if isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.history["loss"] += loss
            self.history["batch_count"] += 1

class Trainer(object):
    def __init__(self, model, nc, optimizer, criterion, device='cpu', scheduler=None, evaluator=None, writer=None, checkpoint_path='.', quiet=True):
        # parameters
        self.model = model
        self.nc = nc
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluator = evaluator
        self.writer = writer
        self.device = device
        self.quiet = quiet
        self.checkpoint_path = checkpoint_path
        
        # set up scheduler
        if scheduler is not None:
            scheduler = Scheduler(scheduler)
        self.scheduler = scheduler
        
        # get model name
        if isinstance(self.model, DataParallel):
            self.model_name = self.model.module.name
        else:
            self.model_name = self.model.name
        
        # history
        self.metrics_history = {}
    
    def train(self, nepochs, dataset, validation_dataset=None, batch_loss_every=4, eval_every=2, 
                    checkpoint_every=None, debug=False, optimizer_kwargs={}, scheduler_kwargs={}):
        # begin training
        if not self.quiet:
            logging.info("Beginning Training ({} epochs)".format(nepochs))
        
        if debug:
            mem_stats = {
                "current": [],
                "peak": [],
                "epoch_start": []
            }
        
        batch_count = 0
        first_epoch = True
        for epoch in range(nepochs):
            # set model to training mode
            self.model.train()
            
            if debug:
                mem_stats['epoch_start'].append(batch_count)
            
            # forward + backward + update
            epoch_loss = 0
            n = 0
            for batch in dataset:
                oom = False
                # update the model weights
                batch, y, mask = processBatch(self.device, batch)
                
                # check for OOM errors
                try:
                    loss = self.optimizer_step(batch, y, mask, **optimizer_kwargs)
                except RuntimeError as e: # out of memory
                    logging.info("Runtime error -- skipping batch.")
                    logging.debug("Error at loss computation.", exc_info=e)
                    oom = True
                if oom:
                    continue
                
                if debug:
                    mem_stats['peak'].append(torch.cuda.max_memory_allocated(self.device)/1e6)
                    mem_stats['current'].append(torch.cuda.memory_allocated(self.device)/1e6)
                    
                # update scheduler
                if self.scheduler is not None:
                    self.scheduler.step(epoch, loss, **scheduler_kwargs)
                
                # write batch-level stats
                if batch_count % batch_loss_every  == 0:
                    if(self.writer):
                        self.writer.add_scalar("train/batch_loss", loss, batch_count)
                
                # update batch count
                batch_count += 1
                epoch_loss += loss
                n += 1
            
            epoch = epoch+1
            # compute metrics
            if (epoch % eval_every == 0) and (self.evaluator is not None):
                metrics = {}
                metrics['train'] = self.evaluator.getMetrics(dataset, use_mask=True, report_threshold=True)
                metrics['train']['loss'] = epoch_loss/(n + 1e-5)
                
                if(validation_dataset is not None):
                    metrics['val'] = self.evaluator.getMetrics(validation_dataset, use_mask=True)
                
                # report performance
                if(not self.quiet):
                    reportMetrics(metrics,
                        label=('Epoch', epoch),
                        header=first_epoch
                    )
                self.updateHistory(metrics, epoch)
                first_epoch = False
            
            # checkpoint
            if checkpoint_every and (epoch % checkpoint_every == 0):
                fname = ospj(self.checkpoint_path, "{}.{}.tar".format(self.model_name, epoch))
                logging.info("Writing checkpoint to file {} at epoch {}".format(fname, epoch))
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch
                        #'model_parameters': self.model.params() ## implement soon
                    }, fname)
        
        if debug:
            return mem_stats
    
    def optimizer_step(self, batch, y, mask=None, use_mask=True, use_weight=True, weight=None):
        # decide how to weight classes
        if use_weight and weight is None:
            weight = classWeights(y, self.nc, device=self.device)
        elif not use_weight:
            weight = None
        
        self.optimizer.zero_grad()
        output = self.model(batch)
        
        # decide if we mask some vertices
        if use_mask:
            loss = self.criterion(output[mask], y[mask], weight=weight)
        else:
            loss = self.criterion(output, y, weight=weight)
        
        # compute gradients
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def updateHistory(self, metrics, epoch):
        for tag in metrics:
            if tag not in self.metrics_history:
                self.metrics_history[tag] = {}
            
            for metric in metrics[tag]:
                # add metric to Tensorboard writer
                if(self.writer):
                    self.writer.add_scalar("{}/{}".format(tag, metric), metrics[tag][metric], epoch)
                
                # add metric to history
                if metric not in self.metrics_history[tag]:
                    self.metrics_history[tag][metric] = []
                self.metrics_history[tag][metric].append(metrics[tag][metric])
