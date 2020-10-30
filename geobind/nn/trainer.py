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
    def __init__(self, model, nc, optimizer, criterion, 
            device='cpu', scheduler=None, evaluator=None, 
            writer=None, checkpoint_path='.', quiet=True
    ):
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
        
        # variables to track training progress
        self.best_state = None
        self.best_state_metric = None
        self.best_epoch = 0
        
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
    
    def train(self, nepochs, dataset,
        validation_dataset=None, batch_loss_every=4, eval_every=2, debug=False,
        checkpoint_every=None, optimizer_kwargs={}, scheduler_kwargs={},
        best_state_metric=None, best_state_metric_threshold=None, 
        best_state_metric_dataset='validation', best_state_metric_goal='max'
    ):
        # begin training
        if not self.quiet:
            logging.info("Beginning Training ({} epochs)".format(nepochs))
        
        if debug:
            mem_stats = {
                "current": [],
                "peak": [],
                "epoch_start": []
            }
        
        if best_state_metric_goal == 'max':
            self.best_state_metric = -999999
        else:
            self.best_state_metric = 999999
        
        batch_count = 0
        first_epoch = True
        for epoch in range(nepochs):
            # set model to training mode
            self.model.train()
            
            # forward + backward + update
            epoch_loss = 0
            n = 0
            for batch in dataset:
                oom = False
                # update the model weights
                batch_data = processBatch(self.device, batch)
                batch, y, mask = batch_data['batch'], batch_data['y'], batch_data['mask']
                
                # check for OOM errors
                try:
                    loss = self.optimizer_step(batch, y, mask, **optimizer_kwargs)
                except RuntimeError as e: # out of memory
                    logging.info("Runtime error -- skipping batch.")
                    logging.debug("Error at loss computation.", exc_info=e)
                    oom = True
                if oom:
                    continue
                    
                # update scheduler
                if self.scheduler is not None:
                    self.scheduler.step(epoch, loss, **scheduler_kwargs)
                
                # write batch-level stats
                if batch_count % batch_loss_every  == 0:
                    if self.writer:
                        self.writer.add_scalar("train/batch_loss", loss, batch_count)
                
                ######### adding scalar paramters of model to tensorboard ######
                if((params_to_write is not None) and self.writer):
                    for name, param in self.model.named_parameters():
                        for param_to_write in params_to_write:
                            if ((param_to_write in name) and param.requires_grad):
                                self.writer.add_scalar(name, param.data.cpu().numpy()[0], self.batch_count)
                                self.writer.add_scalar(name + "_grad", param.grad.cpu().numpy()[0], self.batch_count)
                               #print (name, param.data, param.requires_grad, param.grad)
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
                    metrics['validation'] = self.evaluator.getMetrics(validation_dataset, use_mask=True)
                
                # report performance
                if(not self.quiet):
                    reportMetrics(metrics,
                        label=('Epoch', epoch),
                        header=first_epoch
                    )
                self.updateHistory(metrics, epoch)
                first_epoch = False
                
                if best_state_metric:
                    state_metric = metrics[best_state_metric_dataset][best_state_metric]
                    if best_state_metric_goal == 'max' and state_metric > best_state_metric_threshold:
                        if state_metric > self.best_state_metric:
                            self.best_state_metric = state_metric
                            self.best_state = self.model.state_dict()
                            self.best_epoch = epoch
                    elif best_state_metric_goal == 'min' and state_metric < best_state_metric_threshold:
                        if state_metric < self.best_state_metric:
                            self.best_state_metric = state_metric
                            self.best_state = self.model.state_dict()
                            self.best_epoch = epoch
                
            # checkpoint
            if checkpoint_every and (epoch % checkpoint_every == 0):
                fname = ospj(self.checkpoint_path, "{}.{}.tar".format(self.model_name, epoch))
                logging.info("Writing checkpoint to file {} at epoch {}".format(fname, epoch))
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch
                    }, fname)
        
        self.endTraining()
    
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
                    self.metrics_history[tag][metric] = {}
                self.metrics_history[tag][metric][epoch] = metrics[tag][metric]
    
    def endTraining(self):
        """Stuff we want to do at the end of training"""
        logging.info("Training ended.")
        
        # Save best state to file if we kept it
        if self.best_state is not None:
            fname = ospj(self.checkpoint_path, "{}.{}.tar".format(self.model_name, "best"))
            logging.info("Writing best state to file {} (epoch: {})".format(fname, self.best_epoch))
            logging.info("Best tracked metric achieved: {:.3f}".format(self.best_state_metric))
            torch.save({
                    'model_state_dict': self.best_state
                }, fname)
            
