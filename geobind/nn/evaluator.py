# built-in modules
import warnings

# third party modules
import torch
import numpy as np

# geobind modules
from geobind.nn import processBatch
from geobind.nn.metrics import auroc, auprc, balanced_accuracy_score, recall_score, brier_score_loss, specificity
from geobind.nn.metrics import precision_score, jaccard_score, f1_score, accuracy_score, matthews_corrcoef
from geobind.nn.metrics import reportMetrics, chooseBinaryThreshold, meshLabelSmoothness

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
    'smoothness': meshLabelSmoothness,
    'specificity': specificity
}

def registerMetric(name, fn):
    METRICS_FN[name] = fn

class Evaluator(object):
    def __init__(self, model, nc, device="cpu", metrics=None, post_process=None, negative_class=0, labels=None):
        self.model = model # must implement the 'forward' method
        self.device = device
        self.negative_class = negative_class
        
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
                    #'balanced_accuracy': {},
                    #'matthews_corrcoef': {},
                    'mean_iou': {'average': 'macro', 'labels': labels, 'zero_division': 0},
                    'precision': {'average': 'macro', 'zero_division': 0, 'labels': labels},
                    'recall': {'average': 'macro', 'zero_division': 0, 'labels': labels},
                    'accuracy': {},
                    #'auroc': {'average': 'macro'},
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
            else:
                # TODO - assume regression
                pass
            self.metrics = metrics
            self.metrics_check = metrics_check
        else:
            if not isinstance(metrics, dict):
                raise ValueError("The argument 'metrics' must be a dictionary of kwargs and metric names or 'none'!")
            self.metrics = metrics
    
    @torch.no_grad()
    def eval(self, dataset, 
            eval_mode=True,
            batchwise=False,
            use_mask=True,
            return_masks=False,
            return_predicted=False,
            return_batches=True,
            xtras=None,
            split_batches=False,
            **kwargs
        ):
        """Returns numpy arrays!!!"""
        
        def _loop(batch, data_items, y_gts, y_prs, outps, logits, masks, batches):
            batch_data = processBatch(self.device, batch, xtras=xtras)
            batch, y, mask = batch_data['batch'], batch_data['y'], batch_data['test_mask']
            logit = self.model(batch)
            if use_mask:
                y = y[mask].cpu().numpy()
                out = self.post(logit[mask]).cpu().numpy()
                logit = logit[mask].cpu().numpy()
            else:
                y = y.cpu().numpy()
                out = self.post(logit).cpu().numpy()
                logit = logit.cpu().numpy()
            
            y_gts.append(y)
            outps.append(out)
            logits.append(logit)
            if return_masks:
                masks.append(mask.cpu().numpy())
            
            if return_predicted:
                y_prs.append(self.predictClass(out, y, **kwargs))
            
            if xtras is not None:
                # these items will not be masked even if `use_mask == True`
                for item in xtras:
                    data_items[item].append(batch_data[item].cpu().numpy())
            
            if return_batches:
                if isinstance(batch, list):
                    if batchwise:
                        batches.append(batch)
                    else:    
                        batches += batch
                else:
                    if batchwise:
                        batches.append([batch])
                    else:
                        batches.append(batch.to('cpu'))
        
        # eval or training
        if eval_mode:
            self.model.eval()
        
        # evaluate model on given dataset
        data_items = {}
        y_gts = []
        y_prs = []
        outps = []
        logis = []
        masks = []
        batches = []
        if xtras is not None:
            # these items will not be masked even if `use_mask == True`
            for item in xtras:
                data_items[item] = []
        
        # loop over dataset
        for batch in dataset:
            if split_batches:
                dl = batch.to_data_list()
                for d in dl:
                    _loop(d, data_items, y_gts, y_prs, outps, logis, masks, batches)
            else:
                _loop(batch, data_items, y_gts, y_prs, outps, logis, masks, batches)
        
        # decide what to do with each data item
        data_items['y'] = y_gts
        data_items['output'] = outps
        data_items['logits'] = logis
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
        
        # add batches if requested
        if return_batches:
            data_items['batches'] = batches
        
        return data_items
    
    def getMetrics(self, *args, 
            eval_mode=True, 
            metric_values=None, 
            threshold=None, 
            threshold_metric='balanced_accuracy', 
            report_threshold=False,
            metrics_calculation="total",
            split_batches=False,
            use_mask=True,
            label_type="vertex",
            **kwargs
        ):
        if self.metrics is None:
            return {}
        if metric_values is None:
            metric_values = {key: [] for key in self.metrics}
        if 'threshold' in metric_values:
            threshold = metric_values['threshold']
        
        if metrics_calculation == "total":
            batchwise = False
        elif metrics_calculation == "average_batches":
            batchwise = True
        else:
            raise ValueError("Invalid option for `metrics_calculation`: {}".format(metrics_calculation))
        
        # Determine what we were given (a dataset or labels/predictions)
        if len(args) == 1:
            evald = self.eval(args[0], eval_mode=eval_mode, use_mask=False, return_masks=True, return_batches=True, batchwise=batchwise, split_batches=split_batches, **kwargs)
            if batchwise:
                y_gt = evald['y']
                outs = evald['output']
                batches = evald['batches']
                masks = evald['masks']
            else:
                y_gt = [evald['y']]
                outs = [evald['output']]
                batches = [evald['batches']]
                masks = [evald['masks']]
        else:
            if len(args) == 3:
                y_gt, outs, masks = args
                batches = None
            else:
                y_gt, outs, masks, batches = args
                batches = [batches]
            y_gt = [y_gt]
            outs = [outs]
            masks = [masks]
        
        # Get predicted class labels
        nan = float('nan')
        for i in range(len(y_gt)):
            if label_type == "graph":
                y_gt[i] = y_gt[i].flatten()
            y_pr = self.predictClass(outs[i], y_gt[i], metric_values, threshold=threshold, threshold_metric=threshold_metric, report_threshold=report_threshold)
            
            #if batches is not None:
                #y_gt[~masks] = self.negative_class
                #self.getGraphMetrics(batches[i], y_gt[i], y_pr, metric_values)
            
            if masks is not None and use_mask:    
                y_gt[i] = y_gt[i][masks[i]]
                y_pr = y_pr[masks[i]]
                outs[i] = outs[i][masks[i]]
            
            # Compute metrics
            ngt = np.bincount(y_gt[i], minlength=2)
            npr = np.bincount(y_pr, minlength=2)
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
                elif metric == 'smoothness':
                    # use `getGraphMetrics` for this
                    continue
                else:
                    if self.metrics_check[metric](ngt, npr):
                        metric_values[metric].append(METRICS_FN[metric](y_gt[i], y_pr, **kw))
                    else:
                        metric_values[metric].append(nan)
        with warnings.catch_warnings():
            # ignore empty-slice warnings from numpy
            warnings.simplefilter("ignore")
            for key in metric_values:
                metric_values[key] = np.nanmean(metric_values[key])
        
        return metric_values
    
    def getGraphMetrics(self, batches, y_gt, y_pr, metric_values=None):
        if self.metrics is None:
            return {}
        if metric_values is None:
            metric_values = {}
        
        smooth_gt = []
        smooth_pr = []
        ptr = 0
        if not isinstance(batches, list):
            batches = [batches]
        for batch in batches:
            edge_index = batch.edge_index.cpu().numpy()
            slc = slice(ptr, ptr + batch.num_nodes)
            if "smoothness" in self.metrics:
                smooth_gt.append(METRICS_FN["smoothness"](y_gt[slc], edge_index, **self.metrics['smoothness']))
                smooth_pr.append(METRICS_FN["smoothness"](y_pr[slc], edge_index, **self.metrics['smoothness']))
            ptr += batch.num_nodes
        
        if "smoothness" in self.metrics:
            smooth_pr = np.mean(smooth_pr)
            smooth_gt = np.mean(smooth_gt)
            metric_values['smoothness'].append(smooth_pr)
            metric_values['smoothness_relative'].append((smooth_pr - smooth_gt)/smooth_gt)
        
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
