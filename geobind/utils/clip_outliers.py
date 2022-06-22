# third party packages
import numpy as np

def clipOutliers(data, method="IQR", axis=None, impute_nans=True):
    if method == "z-score":
        try:
            from scipy.stats import zscore
        except ModuleNotFoundError:
            raise ModuleNotFoundError("The dependency 'SciPy' is required for this functionality!'")
        Z = np.abs(zscore(data, axis=axis))
        mask = (Z > 3) + np.isnan(data)
    elif method == "IQR":
        Q1 = np.nanquantile(data, 0.25, axis=axis)
        Q3 = np.nanquantile(data, 0.75, axis=axis)
        IQR = Q3 - Q1
        mask = (data < (Q1-1.5*IQR)) + (data > (Q3+1.5*IQR)) + np.isnan(data)
    mdata = np.ma.array(data, mask=mask)
    lower = np.ma.getdata(mdata.min(axis=axis))
    upper = np.ma.getdata(mdata.max(axis=axis))
    
    data = np.clip(data, lower, upper)
    
    if impute_nans:
        inds = np.where(np.isnan(data))
        mean = np.nanmean(data, axis=axis)
        
        if data.ndim == 1:
            data[inds] = mean
        else:
            data[inds] = np.take(mean, inds[1])
    
    return data
