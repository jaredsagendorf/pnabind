# third party modules
import numpy as np

def chooseBinaryThreshold(y_gt, probs, metric_fn,
        mask=None,
        metric_criteria='max',
        n_samples=25,
        optimize="batch_mean",
        **kwargs
    ):
    """ Choose a threshold value which meets the following criteria:
        y_gt: ...
        probs: ...
        criteria (string):
            min - minimizes the score
            max - maximizes the score
        beta (float): the beta value to use when score=F-beta
    """
    # sample thresholds
    thresholds = np.linspace(0, 1, n_samples+2)[1:-1] # skip 0 and 1 values
    m = lambda t: metric_fn(y, p >= t, **kwargs)
    
    # ensure we are working with list of arrays
    if not isinstance(y_gt, list):
        y_gt = [y_gt]
    if not isinstance(probs, list):
        probs = [probs]
    assert len(y_gt) == len(probs)
    
    if mask is not None:
        if not isinstance(mask, list):
            mask = [mask]
    
    # evaluate metric on each provided array
    values = []
    for i in range(len(y_gt)):
        if mask is not None:
            y = y_gt[i][mask[i]]
            p = probs[i][mask[i]]
        else:
            y = y_gt[i]
            p = probs[i]
        if p.ndim == 2:
            p = p[:,1]
        values.append(np.array(list(map(m, thresholds))))
    values = np.array(values).reshape(len(y_gt), len(thresholds))
    
    # decide what to optimize
    if optimize == "batch_mean":
        values = values.mean(axis=0).reshape(-1, len(thresholds))
    elif optimize == "batch_max":
        values = values.max(axis=0).reshape(-1, len(thresholds))
    elif optimize == "batch":
        pass
    else:
        raise ValueError("unrecognized value of kwarg `optimize`")
    
    # choose how to evaluate
    if metric_criteria == 'max':
        idx = np.argmax(values, axis=1)
    elif metric_criteria == 'min':
        idx = np.argmin(values, axis=1)
    
    return thresholds[idx].squeeze(), values[np.arange(len(values)),idx].squeeze()
