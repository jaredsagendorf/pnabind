# third party modules
import torch

def processBatch(device, batch, xtras=None):
    batch_data = {}
    if isinstance(batch, list):
        batch_data['train_mask'] = torch.cat([data.train_mask for data in batch]).to(device)
        batch_data['test_mask'] = torch.cat([data.test_mask for data in batch]).to(device)
        batch_data['y'] = torch.cat([data.y for data in batch]).to(device)
        if xtras is not None:
            for item in xtras:
                batch_data[item] = torch.cat([getattr(data, item) for data in batch]).to(device)
        batch_data['batch'] = batch
    else:
        batch_data['batch'] = batch.to(device)
        batch_data['train_mask'] = batch.train_mask
        batch_data['test_mask'] = batch.test_mask
        batch_data['y'] = batch.y
        if xtras is not None:
            for item in xtras:
                batch_data[item] = getattr(batch, item)
    
    return batch_data
