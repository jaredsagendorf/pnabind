# third party packages
import numpy as np

def clipOutliers(data, method="IQR", axis=None):
    if method == "z-score":
        try:
            from scipy.stats import zscore
        except ModuleNotFoundError:
            raise ModuleNotFoundError("The dependency 'SciPy' is required for this functionality!'")
        Z = np.abs(zscore(data, axis=axis))
        mask = np.logical_or((Z > 3), np.isnan(data))
    elif method == "IQR":
        Q1 = np.nanquantile(data, 0.25, axis=axis)
        Q3 = np.nanquantile(data, 0.75, axis=axis)
        IQR = Q3 - Q1 
        mask = np.logical_or((data < (Q1-1.5*IQR)), (data > (Q3+1.5*IQR)), np.isnan(data))
    mdata = np.ma.array(data, mask=mask)
    lower = mdata.min(axis=axis)
    upper = mdata.max(axis=axis)
    
    return np.clip(data, lower, upper)
