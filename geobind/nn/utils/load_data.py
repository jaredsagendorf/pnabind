# builtin modules
import os.path as osp
import hashlib
from pickle import dump, load

# third party modules
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose, FaceToEdge, PointPairFeatures
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
    def __init__(self, data_files, nc, data_dir,
            save_dir=None,
            transform=None,
            pre_transform=Compose([FaceToEdge(remove_faces=False), PointPairFeatures(cat=True)]),
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
        self.nc = nc
        self.balance = balance
        self.percentage = percentage
        self.remove_mask = remove_mask
        self.unmasked_class = unmasked_class
        self.scale = scale
        self.scaler = scaler
        self._scaler = None
        
        if(self.scale and self.scaler is None):
            self._scaler = NodeScaler()
        
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
            nc,
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
        data_list = []
        
        # read and process datafiles
        for f in self.raw_paths:
            data_arrays = np.load(f)
            
            if self.remove_mask:
                # remove any previous masking
                data_arrays['Y'][(data_arrays['Y'] == -1)] = self.unmasked_class
            
            if self.balance == 'balanced':
                idxb = balancedClassIndices(data_arrays['Y'], range(self.nc), max_percentage=self.percentage)
            elif self.balance == 'unmasked':
                idxb = (data_arrays['Y'] >= 0)
            elif self.balance == 'all':
                idxb = (data_arrays['Y'] == data_arrays['Y'])
            else:
                raise ValueError("Unrecognized value for `balance` keyword: {}".format(balance))
            
            if self._scaler is not None:
                # add node features to learned scaler
                self._scaler.update(data_arrays['X'][idxb])
            
            data = Data(
                x=torch.tensor(data_arrays['X'], dtype=torch.float32),
                y=torch.tensor(data_arrays['Y'], dtype=torch.int64),
                pos=torch.tensor(data_arrays['V'], dtype=torch.float32),
                norm=torch.tensor(data_arrays['N'], dtype=torch.float32),
                face=torch.tensor(data_arrays['F'].T, dtype=torch.int64),
                edge_attr=None,
                edge_index=None
            )
            data.mask = torch.tensor(idxb, dtype=torch.bool)
            data_list.append(data)
        
        if self.scale and (self.scaler is None):
            # set the scaler to the learned scaler
            self.scaler = self._scaler
            self.scaler.fit()
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
                
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        if self.scale:
            # scale node features in each data object
            for data in data_list:
                data.x = torch.tensor(self.scaler.scale(data.x), dtype=torch.float32)
        
        # save data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # save scaler
        dump(self.scaler, open(self.processed_paths[1], "wb"))
