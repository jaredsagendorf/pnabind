# built-in modules
import warnings

# third party modules
import torch
import numpy as np

# geobind modules
from geobind.nn import processBatch
from geobind.nn.metrics import auroc, auprc, balanced_accuracy_score, recall_score, brier_score_loss, specificity
from geobind.nn.metrics import precision_score, jaccard_score, f1_score, accuracy_score, matthews_corrcoef
from geobind.nn.metrics import reportMetrics, chooseBinaryThreshold#, meshLabelSmoothness

METRICS_FN = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'mean_iou': jaccard_score,
    'auroc': auroc,
    'auprc': auprc,
    'recall': recall_score,
    'precision': precision_score,
    'f1_score': f1_score,
    'brier_score': brier_score_loss,
    'matthews_corrcoef': matthews_corrcoef,
    #'smoothness': meshLabelSmoothness,
    'specificity': specificity
}

def registerMetric(name, fn):
    METRICS_FN[name] = fn

class Evaluator(object):
    def __init__(self, model, nc, device="cpu", metrics=None, post_process=None, negative_class=0, labels=None):
        self.model = model # must implement the 'forward' method
        self.device = device
        self.negative_class = negative_class
        self.nc = nc
        
        if post_process is None:
            # identity function
            post_process = lambda x: x
        self.post = post_process
        
        # decide what metrics to use
        self.nc = nc
        if metrics == 'none':
            self.metrics = None
        elif metrics is None:
            if nc == 2:
                # binary classifier
                metrics={
                    'auroc': {'average': 'binary'},
                    'auprc': {'average': 'binary'},
                    'balanced_accuracy': {},
                    'mean_iou': {'average': 'weighted'},
                    'precision': {'average': 'binary', 'zero_division': 0},
                    'recall': {'average': 'binary', 'zero_division': 0},
                    'accuracy': {},
                    'specificity': {},
                    'matthews_corrcoef': {}
                }
                metrics_check = {
                    'auroc': lambda n1, n2: (n1[0] > 0) and (n1[1] > 0),
                    'auprc': lambda n1, n2: (n1[0] > 0) and (n1[1] > 0),
                    'balanced_accuracy': lambda n1, n2: (n1[0] > 0) and (n1[1] > 0),
                    'mean_iou': lambda n1, n2: (n1[0] > 0) and (n1[1] > 0),
                    'precision': lambda n1, n2: (n1[0] > 0) and (n1[1] > 0),
                    'recall': lambda n1, n2: (n1[1] > 0),
                    'accuracy': lambda n1, n2: True,
                    'specificity': lambda n1, n2: (n1[0] > 0),
                    'matthews_corrcoef': lambda n1, n2 : (n1[0] > 0) and (n1[1] > 0) and (n2[0] > 0) and (n2[1] > 0)
                }
            elif nc > 2:
                # three or more classes 
                if labels is None:
                    labels = list(range(nc))
                    labels.remove(negative_class)
                metrics={
                    'matthews_corrcoef': {},
                    'mean_iou': {'average': 'macro', 'labels': labels, 'zero_division': 0},
                    'precision': {'average': 'macro', 'zero_division': 0, 'labels': labels},
                    'recall': {'average': 'macro', 'zero_division': 0, 'labels': labels},
                    'accuracy': {},
                }
                metrics_check = {
                    'auroc': None,
                    'auprc': None,
                    'balanced_accuracy': None,
                    'mean_iou': None,
                    'precision': None,
                    'recall': None,
                    'accuracy': None,
                    'specificity': None
                }
            self.metrics = metrics
            self.metrics_check = metrics_check
        else:
            if not isinstance(metrics, dict):
                raise ValueError("The argument 'metrics' must be a dictionary of kwargs and metric names or 'none'!")
            self.metrics = metrics
    
    @torch.no_grad()
    def eval(self, dataset, 
            eval_mode=True,
            return_masks=True,
            return_batches=False,
            xtras=None,
            split_batches=False,
            forward_kwargs={},
            **kwargs
        ):
        """Returns numpy arrays!!!"""
       
        def _to_array(data):
            if isinstance(data, tuple):
                return tuple([d.cpu().numpy() for d in data])
            else:
                return data.cpu().numpy()
            
        def _loop(batch, return_mask=True, return_batch=False, xtras=None, **kwargs):
            batch_data = processBatch(self.device, batch, xtras=xtras)
            batch, y, mask = batch_data['batch'], batch_data['y'], batch_data['test_mask']
            output = self.model(batch, **kwargs)
            
            return_data = {}
            return_data['output'] = _to_array(output) # array or tuple
            return_data['y'] = y.cpu().numpy()
            if return_masks:
                return_data['mask'] = mask.cpu().numpy()
            
            if return_batch:
                return_data['batch'] = batch.to('cpu')
            
            if xtras is not None:
                for item in xtras:
                    return_data[item] = batch_data[item].cpu().numpy()
            
            return return_data
        
        # eval or training
        if eval_mode:
            self.model.eval()
        
        # evaluate model on given dataset
        data_items = []
        for batch in dataset:
            if split_batches:
                dl = batch.to_data_list()
                for d in dl:
                    data_items.append(_loop(d, return_mask=return_masks, return_batch=return_batches, xtras=xtras, **forward_kwargs))
            else:
                data_items.append(_loop(batch, return_mask=return_masks, return_batch=return_batches, xtras=xtras, **forward_kwargs))
        
        return data_items
    
    def getMetrics(self, *args, 
            eval_mode=True,
            metric_values=None, 
            threshold=None, 
            threshold_metric='balanced_accuracy',
            report_threshold=True,
            metrics_calculation="macro",
            split_batches=False,
            use_masks=True,
            label_type="vertex",
            report_statistic="mean",
            **kwargs
        ):
        
        # Set up various metric criteria
        if metric_values is None:
            if self.metrics is None:
                # no information given on what metrics to calculate
                return {}
            metric_values = {key: [] for key in self.metrics}
        if 'threshold' in metric_values:
            threshold = metric_values['threshold']
        
        if metrics_calculation == "macro":
            # flatten all batches into single array
            batchwise = False
            split_batches = False
        elif metrics_calculation == "micro":
            # compute average over batches
            batchwise = True
        else:
            raise ValueError("Invalid option for `metrics_calculation`: {}".format(metrics_calculation))
        
        # Determine what we were given (a dataset or labels/predictions)
        if len(args) == 1:
            if isinstance(args[0], list):
                data_items = args[0]
            else:
                # assume we were passed a dataloader object
                DL = args[0]
                data_items = self.eval(DL, eval_mode=eval_mode, return_masks=use_masks, return_batches=False, split_batches=split_batches, **kwargs)
            
            # extract arrays from data items
            if use_masks:
                y_gt = [d['y'][d['mask']] for d in data_items]
                outs = [self.post(d['output'][d['mask']]) for d in data_items]
            else:
                y_gt = [d['y'] for d in data_items]
                outs = [self.post(d['output']) for d in data_items]
            
            if not batchwise:
                # flatten everything into single array
                y_gt = [np.concatenate(y_gt)]
                outs = [np.concatenate(outs)]
        else:
            y_gt, outs = args
            # make sure we have list of arrays
            if not isinstance(y_gt, list):
                y_gt = [y_gt]
            if not isinstance(outs, list):
                outs = [outs]
        
        # Check if threshold was set
        if threshold is None:
            threshold, _ = chooseBinaryThreshold(y_gt, outs, METRICS_FN[threshold_metric], **self.metrics[threshold_metric])
        
        # Get predicted class labels
        nan = float('nan')
        for i in range(len(y_gt)):
            if label_type == "graph":
                y_gt[i] = y_gt[i].flatten()
            y_pr = self.predictClass(outs[i], y_gt[i], metric_values, threshold=threshold, threshold_metric=threshold_metric, report_threshold=report_threshold)
            
            # Compute metrics
            ngt = np.bincount(y_gt[i], minlength=self.nc)
            npr = np.bincount(y_pr, minlength=self.nc)
            if self.nc > 2:
                y_gt[i] = np.eye(self.nc)[y_gt[i]]
                y_pr = np.eye(self.nc)[y_pr]
            
            for metric, kw in self.metrics.items():
                if metric == 'auprc' or metric == 'auroc':
                    # AUC metrics
                    if self.metrics_check[metric](ngt, npr):
                        metric_values[metric].append(METRICS_FN[metric](y_gt[i], outs[i], **kw))
                    else:
                        metric_values[metric].append(nan)
                else:
                    if self.metrics_check[metric](ngt, npr):
                        metric_values[metric].append(METRICS_FN[metric](y_gt[i], y_pr, **kw))
                    else:
                        metric_values[metric].append(nan)
        
        # Determine how aggregate metrics values
        if report_statistic is not None:
            with warnings.catch_warnings():
                # ignore empty-slice warnings from numpy
                warnings.simplefilter("ignore")
                for key in metric_values:
                    if report_statistic == "mean":
                        metric_values[key] = np.nanmean(metric_values[key])
                    elif report_statistic == "max":
                        metric_values[key] = np.nanmax(metric_values[key])
                    elif report_statistic == "min":
                        metric_values[key] = np.nanmin(metric_values[key])
                    elif report_statistic == "median":
                        metric_values[key] = np.nanmedian(metric_values[key])
        
        return metric_values
    
    def predictClass(self, p, y_gt=None, metrics_dict=None, threshold=None, threshold_metric='balanced_accuracy', report_threshold=False):
        # Decide how to determine `y_pr` from `outs`
        if self.nc == 2:
            if (threshold is None) and (y_gt is not None):
                # sample n_samples threshold values
                threshold, _  = chooseBinaryThreshold(y_gt, p[:,1], METRICS_FN[threshold_metric], **self.metrics[threshold_metric])
            elif (threshold is None) and (y_gt is None):
                threshold = 0.5
            y_pr = (p[:,1] >= threshold)
            
            if report_threshold and (metrics_dict is not None):
                metrics_dict['threshold'] = threshold
        elif self.nc > 2:
            y_pr = np.argmax(p, axis=1).flatten()
        
        return y_pr
    
    # def getGraphMetrics(self, batches, y_gt, y_pr, metric_values=None):
        # if self.metrics is None:
            # return {}
        # if metric_values is None:
            # metric_values = {}
        
        # smooth_gt = []
        # smooth_pr = []
        # ptr = 0
        # if not isinstance(batches, list):
            # batches = [batches]
        # for batch in batches:
            # edge_index = batch.edge_index.cpu().numpy()
            # slc = slice(ptr, ptr + batch.num_nodes)
            # if "smoothness" in self.metrics:
                # smooth_gt.append(METRICS_FN["smoothness"](y_gt[slc], edge_index, **self.metrics['smoothness']))
                # smooth_pr.append(METRICS_FN["smoothness"](y_pr[slc], edge_index, **self.metrics['smoothness']))
            # ptr += batch.num_nodes
        
        # if "smoothness" in self.metrics:
            # smooth_pr = np.mean(smooth_pr)
            # smooth_gt = np.mean(smooth_gt)
            # metric_values['smoothness'].append(smooth_pr)
            # metric_values['smoothness_relative'].append((smooth_pr - smooth_gt)/smooth_gt)
        
        # return metric_values
