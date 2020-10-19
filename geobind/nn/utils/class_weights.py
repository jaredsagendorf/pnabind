# third party modules
import torch

# geobind modules
from geobind.nn import processBatch

def classWeights(data, nc, device='cpu', use_mask=True):
    if isinstance(data, torch.Tensor):
        # a tensor of class labels
        weight = data.shape[0]/(nc*torch.eye(nc)[data].sum(axis=0))
    else:
        # a dataloader object
        ys = []
        for batch in data:
            batch, y, mask = processBatch(device, batch)
            if use_mask :
                y = y[mask]
            ys.append(y)
        
        ys = torch.cat(ys, axis=0)
        weight = ys.shape[0]/(nc*torch.eye(nc)[ys].sum(axis=0))
    
    return weight.to(device)
