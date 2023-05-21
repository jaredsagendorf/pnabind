import numpy as np
from sklearn.base import BaseEstimator

class DummyModel(BaseEstimator):
    def predict_proba(self, X):
        return X
    
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        
        return self
