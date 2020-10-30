# third party packages
import numpy as np
from scipy.stats import zscore

def clipOutliers(data, method="IQR", axis=None):
    if(method == "z-score"):
        Z = np.abs(zscore(data, axis=axis))
        mask = (Z > 3)
    elif(method == "IQR"):
        Q1 = np.quantile(data, 0.25, axis=axis)
        Q3 = np.quantile(data, 0.75, axis=axis)
        IQR = Q3 - Q1 
        mask = np.logical_or((data < (Q1-1.5*IQR)), (data > (Q3+1.5*IQR)))
    mdata = np.ma.array(data, mask=mask)
    lower = mdata.min(axis=axis)
    upper = mdata.max(axis=axis)
    
    return np.clip(data, lower, upper)
