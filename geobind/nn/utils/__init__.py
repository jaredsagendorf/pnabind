from .balanced_class_indices import balancedClassIndices
from .class_weights import classWeights
from .load_data import ClassificationDatasetMemory
from .load_data import loadDataset
from .mlp import MLP
from .load_module import loadModule
from .add_weight_decay import addWeightDecay

__all__ = [
    "balancedClassIndices",
    "classWeights",
    "ClassificationDatasetMemory",
    "loadDataset",
    "loadModule",
    "addWeightDecay",
    "MLP"
]

## standard packages
#import logging
#from os.path import join as ospj

## third party packages
#import numpy as np
#import torch
#import torch.nn.functional as F
#from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, jaccard_score
#from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
#from torch_geometric.nn import DataParallel
#from torch_geometric.data import Data, DataLoader, DataListLoader
#from torch_geometric.transforms import PointPairFeatures, GenerateMeshNormals
#from scipy import sparse
#from sklearn.preprocessing import StandardScaler

## geobind modules

#gmn = GenerateMeshNormals()
#ppf = PointPairFeatures()


#def loadModel(config, nIn=None, nOut=None, tarfile=None):
    #if(nIn is None):
        #nIn = config["num_features"]
    #if(nOut is None):
        #nOut = config["num_classes"]
    
    #if(config["model"]["name"] == "Net_Conv_EdgePool"):
        #import geobind
        #from geobind.nn.models import Net_Conv_EdgePool
        #model = Net_Conv_EdgePool(nIn, nOut, **config["model"]["kwargs"])
    
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #if(tarfile is not None):
        #checkpoint = torch.load(tarfile, map_location=device)
        
        ## DELETE IN FUTURE
        #from collections import OrderedDict
        #new_state_dict = OrderedDict()
        #for k, v in checkpoint["model_state_dict"].items():
            #name = k.replace("module.", "") # remove module
            #new_state_dict[name] = v
        #####################################################
        
        #model.load_state_dict(new_state_dict)
    
    #model.device = device
    #return model


#def computePPEdgeFeatures(data, scale=False, norm_only=False):
    ## generate edge features
    #if(data.norm is None):
        #gmn(data)
    
    #if(not norm_only):
        #ppf(data)
        
        ## scale edge features to lie within [0,1]
        #if(scale == "clip"):
            #e_mean = data.edge_attr.mean(axis=0)
            #e_std = data.edge_attr.std(axis=0)
            #e_min = e_mean - 2*e_std
            #e_max = e_mean + 2*e_std
            #data.edge_attr = torch.clamp((data.edge_attr - e_min)/(e_max - e_min), min=0.0, max=1.0)
        #elif(scale == "norm"):
            #e_mean = data.edge_attr.mean(axis=0)
            #e_std = data.edge_attr.std(axis=0)
            #data.edge_attr = (data.edge_attr - e_mean)/e_std




