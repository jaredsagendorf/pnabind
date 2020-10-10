# third party packages
import torch
import torch.nn as nn
import torch.nn.functional as F


# geobind modules
from geobind.nn.models import NetConvEdgePool, PointNetPP

class MultiBranchNet(torch.nn.Module):
    def __init__(self, nIn, nOut, 
            nhidden=32,
            act='relu',
            kwargs1=None,
            kwargs2=None,
            name='multi_branch_net'
    ):
        super(MultiBranchNet, self).__init__()
        if(act == 'relu'):
            self.act = F.relu
        elif(act == 'elu'):
            self.act = F.elu
        elif(act == 'selu'):
            self.act = F.selu
        self.name = name
        
        self.branch1 = PointNetPP(nIn, lin=False, nhidden=nhidden, **kwargs1)
        self.branch2 = NetConvEdgePool(nIn, lin=False, nhidden=nhidden, **kwargs2)
        
        self.lin1 = nn.Linear(2*nhidden, nhidden)
        self.lin2 = nn.Linear(nhidden, nhidden)
        self.lin3 = nn.Linear(nhidden, nOut)
    
    def forward(self, data):
        
        x1 = self.branch1.forward(data)
        x2 = self.branch2.forward(data)
        
        x = torch.cat([x1, x2], axis=-1)
        
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.lin3(x)
        
        return x
