# third party modules
import torch
from torch_geometric.nn import radius, fps

class SAModule(torch.nn.Module):
    def __init__(self, conv, ratio, radius, max_neighbors=32):
        super(SAModule, self).__init__()
        """This module acts as a pooling/conv layer. Taken from pytorch-geometric examples with modifications."""
        self.ratio = ratio
        self.r = radius
        self.K = max_neighbors
        self.conv = conv
    
    def forward(self, x, pos, batch, norm=None, face=None):
        # pool points based on FPS algorithm, returning num_points*ratio centroids
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        
        # finds points within radius `self.r` of the centroids, up to `self.K` pts per centroid
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.K)
        
        # perform convolution over edges joining centroids to their neighbors within ball of radius `self.r`
        row = idx[row]
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, pos, norm, edge_index, faces=face)[idx]
        pos, batch = pos[idx], batch[idx]
        
        return x, pos, batch, idx
