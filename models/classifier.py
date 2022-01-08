import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Dropout
from torch_geometric.nn import radius, fps, knn_interpolate
from torch_geometric.nn import PPFConv, BatchNorm
from torch_geometric.nn import GlobalAttention, global_mean_pool

from geobind.nn.utils import MLP


class SAModule(torch.nn.Module):
    def __init__(self, nIn, nOut, ratio, radius, aggr="max",
            max_neighbors=32, batch_norm=True, dropout=0.0
        ):
        super(SAModule, self).__init__()
        """This module acts as a pooling/conv layer. Taken from pytorch-geometric examples."""
        self.ratio = ratio
        self.r = radius
        self.K = max_neighbors
        
        # set up convolution
        dim = nIn + 4
        if batch_norm:
            nn1 = MLP([dim, dim, dim], batch_norm=True)
            nn2 = MLP([dim, nOut], batch_norm=True)
        else:
            nn1 = MLP([dim, dim, dim], batch_norm=False, dropout=[dropout, 0.0], dropout_position="left")
            nn2 = MLP([dim, nOut], batch_norm=False)
        self.conv = PPFConv(nn1, nn2, add_self_loops=False)
        self.conv.aggr = aggr
    
    def forward(self, x, pos, batch, norm=None):
        # pool points based on FPS algorithm, returning Npt*ratio centroids
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        
        # finds points within radius `self.r` of the centroids, up to `self.K` pts per centroid
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=self.K)
        row = idx[row]
        edge_index = torch.stack([col, row], dim=0)
        
        # perform convolution over edges joining centroids to their neighbors within ball of radius `self.r`
        x = self.conv(x, pos, norm, edge_index)[idx]
        pos, batch = pos[idx], batch[idx]
        
        return x, pos, batch, idx

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nIn, nOut, pool_args):
        super(GlobalSAModule, self).__init__()
        
        self.nIn = nIn
        self.nOut = nOut
        self.pool_type = pool_args["name"]
        
        if pool_args["name"] == "global_mean_pool":
            self.pool = global_mean_pool
        elif pool_args["name"] == "global_max_pool":
            self.pool = global_max_pool
        elif pool_args["name"] == "global_add_pool":
            self.pool = global_add_pool
        elif pool_args["name"] == "topk_pool":
            self.pool = TopKPooling(nIn, ratio=pool_args["k"])
        elif pool_args["name"] == "global_attention_pool":
            self.gate_nn = MLP([nIn, nIn, 1], batch_norm=False, dropout=pool_args["dropout"], dropout_position="left")
            self.nn = nn.Identity()
            self.pool = GlobalAttention(self.gate_nn, self.nn)
    
    def forward(self, x, pos, batch, edge_index=None):
        x = self.pool(x, batch)
        
        return x

class Model(torch.nn.Module):
    def __init__(self, nIn, nout=None,
            pool_args=None,
            nhidden=16,
            depth=3,
            ratios=None,
            radii=None,
            max_neighbors=32,
            name='pointnet_pp',
            dropout=0.0,
            aggr='max',
            batch_norm=True
        ):
        super(Model, self).__init__()
        
        ### Set up ###
        self.depth = depth
        self.name = name
        self.dropout = dropout
        
        ### Model Layers ###
        # linear input
        if batch_norm:
            self.lin_in = MLP([nIn, nhidden, nhidden], batch_norm=True)
        else:
            self.lin_in = MLP([nIn, nhidden, nhidden], batch_norm=False, dropout=dropout)
        
        # pooling layers
        self.GP_module = GlobalSAModule(nhidden, nhidden, pool_args)
        self.SA_modules = torch.nn.ModuleList()
        for i in range(depth):
            self.SA_modules.append(
                SAModule(
                    nhidden,
                    nhidden,
                    ratios[i],
                    radii[i], 
                    max_neighbors=max_neighbors,
                    aggr=aggr,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            )
        
        # linear layers
        if batch_norm:
            self.lin_out = MLP(
                [nhidden, nhidden, nout],
                batch_norm=[True, False],
                act=['relu', None]
            )
        else:
            self.lin_out = MLP(
                [nhidden, nhidden, nout],
                dropout=[dropout, 0.0],
                act=['relu', None]
            )
        self.nout = nout
    
    def forward(self, data):
        # lin1
        x = self.lin_in(data.x)
        
        # conv/pooling
        norm = data.norm
        sa_outs = [(x, data.pos, data.batch)]
        for i in range(self.depth):
            args = (*sa_outs[i], norm)
            x, pos, batch, idx = self.SA_modules[i].forward(*args)
            sa_outs.append((x, pos, batch))
            norm = norm[idx]
        
        # global pooling
        x = self.GP_module(*sa_outs[-1])
        
        # lin2
        x = F.dropout(x, self.dropout, self.training)
        x = self.lin_out(x)
        
        return x
