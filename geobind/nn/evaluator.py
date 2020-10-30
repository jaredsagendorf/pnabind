# third party modules
import torch
import numpy as np

# geobind modules
from geobind.nn import processBatch
from geobind.nn.metrics import auroc, auprc, balanced_accuracy_score, recall_score, brier_score_loss
from geobind.nn.metrics import precision_score, jaccard_score, f1_score, accuracy_score, matthews_corrcoef
from geobind.nn.metrics import reportMetrics, chooseBinaryThreshold

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
    'matthews_corrcoef': matthews_corrcoef
}

def registerMetric(name, fn):
    METRICS_FN[name] = fn

class Evaluator(object):
    def __init__(self, model, nc, device="cpu", metrics=None, post_process=None, negative_class=0, labels=None):
        self.model = model # must implement the 'forward' method
        self.device = device
        
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
                    'accuracy': {}
                }
            elif nc > 2:
                # three or more classes 
                if labels is None:
                    labels = list(range(nc))
                    labels.remove(negative_class)
                metrics={
                    'balanced_accuracy': {},
                    'mean_iou': {'average': 'weighted', 'labels': labels},
                    'precision': {'average': 'weighted', 'zero_division': 0, 'labels': labels},
                    'recall': {'average': 'weighted', 'zero_division': 0, 'labels': labels},
                    'accuracy': {},
                    'matthews_corrcoef': {}
                }
            else:
                # TODO - assume regression
                pass
            self.metrics = metrics
        else:
            if not isinstance(metrics, dict):
                raise ValueError("The argument 'metrics' must be a dictionary of kwargs and metric names or 'none'!")
            self.metrics = metrics
    
    @torch.no_grad()
    def eval(self, dataset, batchwise=False, use_mask=True, return_masks=False, return_predicted=False, xtras=None, **kwargs):
        """Returns numpy arrays!!!"""
        self.model.eval()
        
        # evaluate model on given dataset
        data_items = {}
        y_gts = []
        y_prs = []
        outps = []
        masks = []
        if xtras is not None:
            # these items will not be masked even if `use_mask == True`
            for item in xtras:
                data_items[item] = []
        
        # loop over dataset
        for batch in dataset:
            batch_data = processBatch(self.device, batch, xtras=xtras)
            batch, y, mask = batch_data['batch'], batch_data['y'], batch_data['mask']
            output = self.model(batch)
            if use_mask:
                y = y[mask].cpu().numpy()
                out = self.post(output[mask]).cpu().numpy()
            else:
                y = y.cpu().numpy()
                out = self.post(output).cpu().numpy()
            
            y_gts.append(y)
            outps.append(out)
            if return_masks:
                masks.append(mask.cpu().numpy())
            
            if return_predicted:
                y_prs.append(self.predictClass(out, y, **kwargs))
            
            if xtras is not None:
                # these items will not be masked even if `use_mask == True`
                for item in xtras:
                    data_items[item].append(batch_data[item].cpu().numpy())
        
        # decide what to do with each data item
        data_items['y'] = y_gts
        data_items['output'] = outps
        if return_masks:
            data_items['masks'] = masks
        if return_predicted:
            data_items['predicted_y'] = y_prs
        
        # concat batches if not batchwise
        if batchwise:
            data_items['num_batches'] = len(y_gts)
        else:
            for item in data_items:
                data_items[item] = np.concatenate(data_items[item], axis=0)
            data_items['num'] = len(data_items['y'])
        
        return data_items
    
    def getMetrics(self, *args, threshold=None, threshold_metric='balanced_accuracy', report_threshold=False, **kwargs):
        if(self.metrics is None):
            return {}
        metric_values = {}
        
        # Determine what we were given (a dataset or labels/predictions)
        if len(args) == 1:
            evald = self.eval(args[0], **kwargs)
            y_gt = evald['y']
            outs = evald['output']
        else:
            y_gt, outs = args
        
        # Get predicted class labels
        y_pr = self.predictClass(outs, y_gt, metric_values, threshold=threshold, threshold_metric=threshold_metric, report_threshold=report_threshold)
        
        # Compute metrics
        for metric, kw in self.metrics.items():
            if(metric in ['auprc', 'auroc']):
                # AUC metrics
                pass
                #metric_values[metric] = METRICS_FN[metric](y_gt, outs, **kw)
            else:
                metric_values[metric] = METRICS_FN[metric](y_gt, y_pr, **kw)
        
        return metric_values
    
    def predictClass(self, outs, y_gt=None, metrics_dict=None, threshold=None, threshold_metric='balanced_accuracy', report_threshold=False):
        # Decide how to determine `y_pr` from `outs`
        if self.nc == 2:
            if (threshold is None) and (y_gt is not None):
                # sample n_samples threshold values
                threshold, _  = chooseBinaryThreshold(y_gt, outs[:,1], metric_fn=METRICS_FN[threshold_metric], **self.metrics[threshold_metric])
            elif (threshold is None) and (y_gt is None):
                threshold = 0.5
            y_pr = (outs[:,1] >= threshold)
            if report_threshold and (metrics_dict is not None):
                metrics_dict['threshold'] = threshold
        elif self.nc > 2:
            y_pr = np.argmax(outs, axis=1).flatten()
        
        return y_pr
