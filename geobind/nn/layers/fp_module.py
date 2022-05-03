# third party modules
import torch
from torch_geometric.nn import knn_interpolate

class FPModule(torch.nn.Module):
    def __init__(self, nn, k=3):
        super(FPModule, self).__init__()
        """This module acts as an unpooling/interpolation layer. Taken from pytorch-geometric examples."""
        self.k = k
        self.nn = nn
    
    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        
        return x, pos_skip, batch_skip
