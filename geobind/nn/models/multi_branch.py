# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# geobind modules
from geobind.nn.models import NetConvPool, PointNetPP
from geobind.nn.layers import ContinuousCRF
from geobind.nn.utils import MLP

class MultiBranchNet(torch.nn.Module):
    def __init__(self, nIn, nOut, 
            nhidden=16,
            act='relu',
            kwargs1=None,
            kwargs2=None,
            name='multi_branch_net',
            use_crf=False,
            crf_args={}
    ):
        super(MultiBranchNet, self).__init__()
        #if(act == 'relu'):
            #self.act = F.relu
        #elif(act == 'elu'):
            #self.act = F.elu
        #elif(act == 'selu'):
            #self.act = F.selu
        self.name = name
        self.use_crf = use_crf
        
        self.branch1 = PointNetPP(nIn, use_lin=False, **kwargs1)
        self.branch2 = NetConvPool(nIn, use_lin=False, **kwargs2)
        
        self.lin_out = MLP([self.branch1.nout + self.branch2.nout, nhidden, nhidden, nOut], batch_norm=False, act=[act, act, None])
        #self.lin1 = nn.Linear(self.branch1.nout + self.branch2.nout, nhidden)
        #self.lin2 = nn.Linear(nhidden, nhidden)
        #self.lin3 = nn.Linear(nhidden, nOut)
        
        if use_crf:
            self.crf1 = ContinuousCRF(**crf_args)
    
    def forward(self, data):
    
        x1 = self.branch1.forward(data)
        x2 = self.branch2.forward(data)
        
        x = torch.cat([x1, x2], axis=-1)
        
        # lin layers
        x = self.lin_out(x)
        #x = self.act(self.lin1(x))
        #x = self.act(self.lin2(x))
        #x = self.lin3(x)
        
        # crf layer
        if self.use_crf:
            x = self.crf1(x, data.edge_index)
        
        return x
