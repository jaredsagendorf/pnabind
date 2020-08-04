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

#class History(object):
    #def __init__(self):
        #self._data = {}
        
    #def update(self, tag, epoch, y, **kwargs):
        #data = self._data
        
        ## check if we have seen this tag before
        #if(tag not in data):
            #data[tag] = {'y': [[]]}
            #for key in kwargs:
                #data[tag][key] = [[]] # 2D of shape EPOCH x BATCH
        #kwargs['y'] = y
        
        ## add more epoch slots if we need them
        #for key in data[tag]:
            #while(epoch + 1 > len(data[tag][key])):
                #data[tag][key].append([[]])
        
        ## update data
        #for key in data[tag]:
            #data[tag][key][epoch].append(kwargs[key])
    
    #def __getitem__(self, tag, name, epoch=None, batch=None):
        #if(epoch is None):
            #return self._data[tag][name]
        #elif(batch is None):
            #return self._data[tag][name][epoch]
        #else:
            #return self._data[tag][name][epoch][batch]

class Trainer(object):
    def __init__(self, model, optimizer, criterion, scheduler=None, evaluator=None, writer=None, checkpoint_path='.', quiet=True):
        # parameters
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.writer = writer
        
        self.quiet = quiet
        self.checkpoint_path = checkpoint_path
        if isinstance(self.model, DataParallel):
            self.model_name = self.model.module.name
        else:
            self.model_name = self.model.name
        
        # history
        self.batch_count = 0
        self.epoch = 0
        self.new_epoch = False
        self.metrics_history = {}
        self.first_epoch = True
    
    def train(self, nepochs, dataset, validation_dataset=None, batch_loss_every=4, eval_every=2, checkpoint_every=None):
        # begin training
        if not self.quiet:
            logging.info("Beginning Training ({} epochs)".format(nepochs))
        
        for epoch in range(nepochs):
            # set model to training mode
            self.model.train()
            
            # forward + backward + update
            self.new_epoch = True
            batch_losses = []
            for batch in dataset:
                # update the model weights
                batch, y, mask = processBatch(self.model, batch)
                loss = self.optimizer_step(batch, y, mask)
                
                # update scheduler
                self.scheduler_step(loss)
                
                # write batch-level stats
                if self.batch_count % batch_loss_every  == 0:
                    #self.metrics_history.update('train/batch_loss', epoch, loss.item(), x=self.batch_count)
                    if(self.writer):
                        self.writer.add_scalar("train/batch_loss", loss.item(), self.batch_count)
                
                # update batch count
                self.batch_count += 1
                self.new_epoch = False
            
            epoch = epoch+1
            # compute metrics
            if (epoch % eval_every == 0) and (self.evaluator is not None):
                metrics = {}
                metrics['train'] = self.evaluator.getMetrics(dataset, mask=True, report_threshold=True)
                
                if(validation_dataset is not None):
                    metrics['val'] = self.evaluator.getMetrics(validation_dataset, mask=True)
                
                # report performance
                if(not self.quiet):
                    reportMetrics(metrics,
                        label=('Epoch', epoch),
                        header=self.first_epoch
                    )
                self.updateHistory(metrics, epoch)
                self.first_epoch = False
            
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
            
            self.epoch += 1
    
    def optimizer_step(self, batch, y, mask=None, weight=True):
        # decide how to weight classes
        if(weight):
            weight = classWeights(y).to(self.model.device)
        else:
            weight = None
        
        self.optimizer.zero_grad()
        output = self.model(batch)
        
        # decide if we mask vertices
        if(mask is not None):
            loss = self.criterion(output[mask], y[mask], weight=weight)
        else:
            loss = self.criterion(output, y, weight=weight)
        
        # compute gradients
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def scheduler_step(self, batch_loss, **kwargs):
        # step per-batch learning rate schedulers if applicable
        if self.scheduler is not None:
            # per-batch schedules
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
                return
            
            # per-epoch schedules
            if self.new_epoch:
                # we are in a new epoch, update per-epoch schedulers
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    mean_loss = self.history['loss'][-1].mean()
                    self.scheduler.step(mean_loss)
                elif isinstance(self.scheduler, ExponentialLR):
                    self.scheduler.step()
    
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
