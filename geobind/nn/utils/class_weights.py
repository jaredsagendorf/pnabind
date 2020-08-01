# third party modules
import torch

def classWeights(y):
    return y.shape[0]/(2*torch.tensor([(y==0).sum(), (y==1).sum()], dtype=torch.float32))
