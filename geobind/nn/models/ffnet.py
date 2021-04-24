import torch
from torch.nn import Linear as Lin
from geobind.nn.utils import MLP
from geobind.nn.layers import ContinuousCRF

class FFNet(torch.nn.Module):
    def __init__(self, nin, nout=None,
            nhidden=32,
            depth=5,
            name='ff_net',
            use_crf=True,
            crf_kwargs={},
            batch_norm=True,
            bn_kwargs={},
            act='relu',
            dropout=0.0
        ):
        super(FFNet, self).__init__()
        self.name = name
        self.use_crf = use_crf
        self.nin = nin
        
        # depth is the number of linear layers
        if isinstance(nhidden, int):
            h = [nin] + [nhidden]*(depth - (nout is not None))
        else:
            h = nhidden
        
        if isinstance(batch_norm, bool):
            b = [batch_norm]*(depth- (nout is not None))
        else:
            b = batch_norm
        
        if isinstance(act, str):
            a = [act]*(depth - (nout is not None))
        else:
            a = act
        
        if isinstance(dropout, float):
            d = [dropout]*(depth - (nout is not None))
        else:
            d = dropout
        
        if nout:
            h = h + [nout]
            b = b + [False]
            a = a + [None]
            d = d + [0.0]
            self.nout = nout
        else:
            self.nout = h[-1]
        
        self.lin = MLP(h, act=a, batch_norm=b, bn_kwargs=bn_kwargs, dropout=d)
        
        if use_crf:
            # continuous CRF layer
            self.crf = ContinuousCRF(**crf_kwargs)
    
    def forward(self, data):
        x = data.x
        x = self.lin(x)
        
        if self.use_crf:
            x = self.crf(x, data.edge_index)
        
        return x
