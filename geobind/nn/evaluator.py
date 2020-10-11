# third party modules
import torch
import numpy as np

# geobind modules
from geobind.nn import processBatch
from geobind.nn.metrics import auroc, auprc, balanced_accuracy_score, recall_score
from geobind.nn.metrics import precision_score, jaccard_score, f1_score, accuracy_score
from geobind.nn.metrics import reportMetrics, chooseBinaryThreshold

METRICS_FN = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'mean_iou': jaccard_score,
    'auroc': auroc,
    'auprc': auprc,
    'recall': recall_score,
    'precision': precision_score,
    'f1_score': f1_score
}

def registerMetric(name, fn):
    METRICS_FN[name] = fn

class Evaluator(object):
    def __init__(self, model, device="cpu", metrics=None, model_type='binary', post_process=None):
        ##TO DO - ADD METRICS AND KWARGS BASED ON MODEL TYPE
        self.model = model # must implement the 'forward' method
        self.device = device
        
        if(post_process is None):
            # identity function
            post_process = lambda x: x
        self.post = post_process
        
        self.model_type = model_type
        
        # decide what metrics to use
        if(metrics == 'none'):
            self.metrics = None
        elif(metrics is None):
            if(model_type == 'binary'):
                # binary classifier
                metrics={
                    'auroc': {},
                    'auprc': {},
                    'balanced_accuracy': {},
                    'mean_iou': {'average': 'weighted'},
                    'precision': {'average': 'binary'},
                    'recall': {'average': 'binary'},
                    'accuracy': {}
                }
            elif(model_type == 'multiclass'):
                # TODO - three or more classes 
                pass
            else:
                # TODO - assume regression
                pass
            self.metrics = metrics
        else:
            if not isinstance(metrics, dict):
                raise ValueError("The argument 'metrics' must be a dictionary of kwargs and metric names or 'none'!")
            self.metrics = metrics
    
    @torch.no_grad()
    def eval(self, dataset, batchwise=False, use_mask=True, return_masks=False):
        """Returns numpy arrays!!!"""
        self.model.eval()
        
        # evaluate model on given dataset
        y_gts = []
        outs = []
        masks = []
        for batch in dataset:
            batch, y, mask = processBatch(self.device, batch)
            output = self.model(batch)
            if(use_mask):
                y = y[mask]
                out = self.post(output[mask])
            else:
                out = self.post(output)
            
            y_gts.append(y.cpu().numpy())
            outs.append(out.cpu().numpy())
            if(return_masks):
                masks.append(mask.cpu().numpy())
        
        # decide what to do with each data item
        if(batchwise):
            # return per-batch output
            if(return_masks):
                return zip(y_gts, outs, masks)
            else:
                zip(y_gts, outs)
        else:
            # concatenate entire dataset
            if(return_masks):
                return np.concatenate(y_gts, axis=0), np.concatenate(outs, axis=0), np.concatenate(masks, axis=0)
            else:
                return np.concatenate(y_gts, axis=0), np.concatenate(outs, axis=0)
    
    def getMetrics(self, *args, threshold=None, threshold_metric='balanced_accuracy', report_threshold=False, **kwargs):
        if(self.metrics is None):
            return {}
        ## TODO: generalize to handle multi-class, regression
        
        # Determine what we were given (a dataset or labels/predictions)
        if(len(args) == 1):
            y_gt, outs = self.eval(args[0], **kwargs)
        else:
            y_gt, outs = args
        
        # Compute metrics
        metric_values = {}
        if(self.model_type == 'binary'):
            if(threshold is None):
                # sample n_samples threshold values
                threshold, _  = chooseBinaryThreshold(y_gt, outs[:,1], metric_fn=METRICS_FN[threshold_metric], **self.metrics[threshold_metric])
            y_pr = (outs[:,1] >= threshold)
            if(report_threshold):
                metric_values['threshold'] = threshold
        elif(self.model_type == 'multiclass'):
            y_pr = np.argmax(outs, axis=1)
        
        for metric, kw in self.metrics.items():
            if(metric in ['auprc', 'auroc']):
                # AUC metrics
                metric_values[metric] = METRICS_FN[metric](y_gt, outs, **kw)
            else:
                metric_values[metric] = METRICS_FN[metric](y_gt, y_pr, **kw)
        
        return metric_values
