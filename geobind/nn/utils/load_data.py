# builtin modules
import os.path as osp
import hashlib
from pickle import dump, load

# third party modules
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler

# geobind modules
from geobind.nn.utils import balancedClassIndices

class NodeScaler(object):
    def __init__(self):
        self._data_arrays = []
        self.scaler = StandardScaler()
    
    def update(self, array):
        self._data_arrays.append(array)
    
    def fit(self):
        self.scaler.fit(np.concatenate(self._data_arrays, axis=0))
    
    def scale(self, array):
        return self.scaler.transform(array)

class ClassificationDatasetMemory(InMemoryDataset):
    def __init__(self, data_files, nc, labels_key, data_dir,
            save_dir=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            balance='balanced',
            percentage=1.0,
            remove_mask=False,
            unmasked_class=0,
            scale=True,
            scaler=None
        ):
        if(save_dir is None):
            save_dir = data_dir
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.data_files = data_files
        self.labels_key = labels_key
        self.nc = nc
        self.balance = balance
        self.percentage = percentage
        self.remove_mask = remove_mask
        self.unmasked_class = unmasked_class
        self.scale = scale
        self.scaler = scaler
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        
        super(ClassificationDatasetMemory, self).__init__(save_dir, transform, pre_transform, pre_filter)
        # load data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # load scaler
        if self.scale and self.scaler is None:
            self.scaler = load(open(self.processed_paths[1], 'rb'))
        
    @property
    def raw_file_names(self):
        return self.data_files
    
    @property
    def processed_file_names(self):
        m = hashlib.md5()
        args = [
            self.nc,
            self.labels_key,
            self.balance,
            self.percentage,
            self.remove_mask,
            self.unmasked_class,
            self.scale
        ]
        args = "".join([str(_) for _ in args] + list(sorted(self.data_files)))
        m.update(args.encode('utf-8'))
        self.hash_name = m.hexdigest()
        return ['{}.pt'.format(self.hash_name), '{}_scaler.pkl'.format(self.hash_name)]
    
    @property
    def raw_dir(self):
        return self.data_dir

    @property
    def processed_dir(self):
        return self.save_dir
    
    def process(self):
        # get datalist and scaler
        data_list, transforms = _processData(self.raw_paths, self.nc, self.labels_key,
            balance=self.balance, 
            remove_mask=self.remove_mask,
            unmasked_class=self.unmasked_class,
            scaler=self.scaler,
            scale=self.scale,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform
        )
        
        # save data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # save scaler
        scaler = transforms['scaler']
        if scaler is not None:
            self.scaler = scaler
            dump(scaler, open(self.processed_paths[1], "wb"))

def _processData(data_files, nc, labels_key, 
        balance="unmasked",
        remove_mask=False,
        unmasked_class=0,
        scaler=None,
        scale=True,
        transform=None,
        pre_filter=None,
        pre_transform=None
    ):
    data_list = []
    
    # read and process datafiles
    for f in data_files:
        data_arrays = np.load(f)
        
        if remove_mask:
            # remove any previous masking
            data_arrays[labels_key][(data_arrays[labels_key] == -1)] = unmasked_class
        
        if balance == 'balanced':
            idxb = balancedClassIndices(data_arrays[labels_key], range(nc), max_percentage=self.percentage)
        elif balance == 'unmasked':
            idxb = (data_arrays[labels_key] >= 0)
        elif balance == 'all':
            idxb = (data_arrays[labels_key] == data_arrays[labels_key])
        else:
            raise ValueError("Unrecognized value for `balance` keyword: {}".format(balance))
        
        data = Data(
            x=torch.tensor(data_arrays['X'], dtype=torch.float32),
            y=torch.tensor(data_arrays[labels_key], dtype=torch.int64),
            pos=torch.tensor(data_arrays['V'], dtype=torch.float32),
            norm=torch.tensor(data_arrays['N'], dtype=torch.float32),
            face=torch.tensor(data_arrays['F'].T, dtype=torch.int64),
            edge_attr=None,
            edge_index=None
        )
        data.mask = torch.tensor(idxb, dtype=torch.bool)
        data_list.append(data)
    
    # filter data
    if pre_filter is not None:
        data_list = [data for data in data_list if pre_filter(data)]
    
    # transform data
    if pre_transform is not None:
        data_list = [pre_transform(data) for data in data_list]
    
    # scale data
    if scale:
        # build a scaler
        if scaler is None:
            scaler = NodeScaler()
            for data in data_list:
                scaler.update(data.x[data.mask])
            scaler.fit()
        
        # scale node features in each data object
        for data in data_list:
            data.x = torch.tensor(scaler.scale(data.x), dtype=torch.float32)
    
    transforms = {
        "scaler": scaler,
        "transform": transform,
        "pre_transform": pre_transform,
        "pre_filter": pre_filter
    }
    return data_list, transforms

def loadDataset(data_files, nc, labels_key, data_dir, cache_dataset=False, **kwargs):
    if cache_dataset:
        dataset = ClassificationDatasetMemory(data_files, nc, labels_key, data_dir, **kwargs)
        transforms = {
            "scaler": dataset.scaler,
            "transform": dataset.transform,
            "pre_transform": dataset.pre_transform,
            "pre_filter": dataset.pre_filter
        }
        info = {
            "num_features": dataset.num_node_features,
            "num_classes": nc,
            "num_instances": len(dataset)
        }
    else:
        data_files = [osp.join(data_dir, f) for f in data_files]
        
        dataset, transforms = _processData(data_files, nc, labels_key, **kwargs)
        info = {
            "num_features": int(dataset[0].x.shape[1]),
            "num_classes": nc,
            "num_instances": len(dataset)
        }
    
    return dataset, transforms, info
