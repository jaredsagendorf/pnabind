# third party modules
import torch

def processBatch(device, batch):
    if(isinstance(batch, list)):
        mask = torch.cat([data.mask for data in batch]).to(device)
        y = torch.cat([data.y for data in batch]).to(device)
    else:
        batch = batch.to(device)
        mask = batch.mask
        y = batch.y
    
    return batch, y, mask
