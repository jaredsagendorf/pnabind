from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, jaccard_score
from sklearn.metrics import f1_score, accuracy_score

def auprc(y_gt, prob, pi=1, **kwargs):
    # aurpr
    pre_vals, rec_vals, _ = precision_recall_curve(y_gt, prob[:,pi])
    auprc = auc(rec_vals, pre_vals)
    
    return auprc

def auroc(y_gt, prob, pi=1, **kwargs):
    # auroc
    fpr, tpr, _ = roc_curve(y_gt, prob[:,pi])
    auroc = auc(fpr, tpr)
    
    return auroc
