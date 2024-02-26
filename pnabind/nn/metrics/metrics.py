import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, jaccard_score
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, matthews_corrcoef
from sklearn.metrics import average_precision_score, roc_auc_score

def auprc(y_gt, prob, pi=1, average='binary', **kwargs):
    # determine if binary or multiclass
    if average == 'binary':
        if (y_gt == 1).sum() == 0:
            return float('nan')
        if prob.ndim > 1:
            prob = prob[:,pi]
        pre_vals, rec_vals, _ = precision_recall_curve(y_gt, prob)
        auprc = auc(rec_vals, pre_vals)
    else:
        nc = prob.shape[1]
        auprc = average_precision_score(np.eye(nc)[y_gt], prob, average=average, **kwargs)
        
    return auprc

def auroc(y_gt, prob, pi=1, average='binary', **kwargs):
    # determine if binary or multiclass
    if average == 'binary':
        if (y_gt == 1).sum() == 0:
            return float('nan')
        if prob.ndim > 1:
            prob = prob[:,pi]
        fpr, tpr, _ = roc_curve(y_gt, prob)
        auroc = auc(fpr, tpr)
    else:
        #nc = prob.shape[1]
        #auroc = roc_auc_score(np.eye(nc)[y_gt], prob, average=average, **kwargs)
        auroc = roc_auc_score(y_gt, prob, average=average, **kwargs)
    
    return auroc

def specificity(ygt, ypr):
    ni = (ygt == 0)
    tn = (ypr[ni] == 0).sum()
    
    return tn/(ni.sum())
