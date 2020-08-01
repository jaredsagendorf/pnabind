# third party modules
import torch
from torch_geometric.nn import DataParallel

def processBatch(model, batch):
    if(isinstance(model, DataParallel)):
        mask = torch.cat([data.mask for data in batch]).to(model.device)
        y = torch.cat([data.y for data in batch]).to(model.device)
    else:
        batch = batch.to(model.device)
        mask = batch.mask
        y = batch.y
    
    return batch, y, mask
