# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# geobind modules
from geobind.nn.models import NetConvPool, PointNetPP, FFNet
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
            crf_kwargs={},
            dropout=0.0,
            batch_norm=True
    ):
        super(MultiBranchNet, self).__init__()
        self.name = name
        self.use_crf = use_crf
        self.nin = nIn
        self.nout = nOut
        
        self.branch1 = self.makeBranch(kwargs1)
        self.branch2 = self.makeBranch(kwargs2)
        
        self.lin_out = MLP(
                [self.branch1.nout + self.branch2.nout, nhidden, nhidden, nOut],
                batch_norm=[batch_norm, batch_norm, False],
                act=[act, act, None],
                dropout=[dropout, dropout, 0.0]
        )
        
        if use_crf:
            self.crf = ContinuousCRF(**crf_kwargs)
    
    def makeBranch(self, kwargs):
        if kwargs["name"] == "NetConvPool":
            b = NetConvPool(self.nin, use_lin=False, **kwargs)
        elif kwargs["name"] == "PointNetPP":
            b = PointNetPP(self.nin, use_lin=False, **kwargs)
        elif kwargs["name"] == "FFNet":
            b = FFNet(self.nin, **kwargs)
        
        return b 
    
    def forward(self, data):
    
        x1 = self.branch1.forward(data)
        x2 = self.branch2.forward(data)
        
        x = torch.cat([x1, x2], axis=-1)
        
        # lin layers
        x = self.lin_out(x)
                
        # crf layer
        if self.use_crf:
            x = self.crf(x, data.edge_index)
        
        return x
