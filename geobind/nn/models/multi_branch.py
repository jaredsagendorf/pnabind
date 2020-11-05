# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# geobind modules
from geobind.nn.models import NetConvEdgePool, PointNetPP
from geobind.nn.layers import ContinuousCRF

class MultiBranchNet(torch.nn.Module):
    def __init__(self, nIn, nOut, 
            nhidden=32,
            act='relu',
            kwargs1=None,
            kwargs2=None,
            name='multi_branch_net',
            crf=False,
            crf_args={}
    ):
        super(MultiBranchNet, self).__init__()
        if(act == 'relu'):
            self.act = F.relu
        elif(act == 'elu'):
            self.act = F.elu
        elif(act == 'selu'):
            self.act = F.selu
        self.name = name
        self.crf = crf
        
        self.branch1 = PointNetPP(nIn, lin=False, nhidden=nhidden, **kwargs1)
        self.branch2 = NetConvEdgePool(nIn, lin=False, nhidden=nhidden, **kwargs2)
        
        if crf:
            self.crf1 = ContinuousCRF(**crf_args)
        
        self.lin1 = nn.Linear(2*nhidden, nhidden)
        self.lin2 = nn.Linear(nhidden, nhidden)
        self.lin3 = nn.Linear(nhidden, nOut)
    
    def forward(self, data):
        
        x1 = self.branch1.forward(data)
        x2 = self.branch2.forward(data)
        
        x = torch.cat([x1, x2], axis=-1)
        
        # crf layer
        if self.crf:
            x = self.crf1(x, data.edge_index)
        
        # lin layers
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.lin3(x)
        
        return x
