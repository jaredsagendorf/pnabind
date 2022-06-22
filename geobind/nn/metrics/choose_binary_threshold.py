# third party modules
import numpy as np

def chooseBinaryThreshold(y_gt, probs, metric_fn, 
        criteria='max',
        n_samples=25,
        minimize_threshold=False,
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
    
    # evaluate metrics on each provided array
    if not isinstance(y_gt, list):
        y_gt = [y_gt]
    if not isinstance(probs, list):
        probs = [probs]
    assert len(y_gt) == len(probs)
    
    values = []
    for i in range(len(y_gt)):
        y = y_gt[i]
        p = probs[i]
        values.append(np.array(list(map(m, thresholds))))
    values = np.array(values).reshape(len(y_gt), len(thresholds)).mean(axis=0)
    
    # choose how to evaluate
    if criteria == 'max':
        idx = np.argmax(values)
    elif criteria == 'min':
        idx = np.argmin(values)
    
    return thresholds[idx], values[idx]
