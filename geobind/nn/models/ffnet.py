import torch
from torch.nn import Linear as Lin
from geobind.nn.utils import MLP
from geobind.nn.layers import ContinuousCRF

class FFNet(torch.nn.Module):
    def __init__(self, nin, nout,
            nhidden=16,
            name='ff_net',
        ):
        super(FFNet, self).__init__()
        h = [nin, nhidden, nhidden, nhidden, nhidden, nhidden]
        self.name = name
        
        self.lin1 = MLP(h)
        self.lin2 = Lin(nhidden, nout)
        
        # continuous CRF layer
        self.crf = ContinuousCRF()
    
    def forward(self, data):
        x = data.x
        x = self.lin1(x)
        x = self.lin2(x)
        
        x = self.crf(x, data.edge_index)
        
        return x
