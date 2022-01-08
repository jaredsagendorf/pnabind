import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius, fps, knn_interpolate
from torch_geometric.nn import PPFConv, BatchNorm
from torch_geometric.nn import GlobalAttention, global_mean_pool

from geobind.nn.utils import MLP

class FPModule(torch.nn.Module):
    def __init__(self, nIn, nSkip, nOut, k=3, batch_norm=True, bn_kwargs={}):
        super(FPModule, self).__init__()
        """This module acts as an unpooling/interpolation layer. Taken from pytorch-geometric examples."""
        self.k = k
        dim = nIn + nSkip
        self.nn = MLP([dim, dim, nOut], batch_norm=batch_norm, bn_kwargs=bn_kwargs)
    
    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        
        return x, pos_skip, batch_skip

class SAModule(torch.nn.Module):
    def __init__(self, nIn, nOut, ratio, radius, aggr="max",
            max_neighbors=32, batch_norm=True
        ):
        super(SAModule, self).__init__()
        """This module acts as a pooling/conv layer. Taken from pytorch-geometric examples."""
        self.ratio = ratio
        self.r = radius
        self.K = max_neighbors
        
        # set up convolution
        dim = nIn + 4
        nn1 = MLP([dim, dim, dim], batch_norm=batch_norm)
        nn2 = MLP([dim, nOut], batch_norm=batch_norm)
        
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
    def __init__(self, nIn, nOut, dropout=0.0, pool_type="global_mean_pool"):
        super(GlobalSAModule, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        
        if pool_type == "global_mean_pool":
            self.pool = global_mean_pool
        elif pool_type == "global_attention_pool":
            self.gate_nn = MLP([nIn, nIn, 1], batch_norm=False, dropout=[dropout, 0.0])
            self.nn = nn.Identity()
            self.pool = GlobalAttention(self.gate_nn, self.nn)
    
    def forward(self, x, pos, batch, edge_index=None):
        x = self.pool(x, batch)
        
        return x

class Model(torch.nn.Module):
    def __init__(self, nIn,
            nout=[2,3],
            nhidden=16,
            depth=3,
            knn_num=3,
            ratios=None,
            radii=None,
            max_neighbors=32,
            name='pointnet_pp',
            pool_type='global_mean_pool',
            dropout=0.0,
            aggr='max',
            batch_norm=True
        ):
        super(Model, self).__init__()
        
        nOut1, nOut2 = nout
        
        ### Set up ###
        self.depth = depth
        self.name = name
        self.dropout = dropout
        
        ### Model Layers ###
        # linear input
        self.lin_in = MLP([nIn, nhidden, nhidden], batch_norm=batch_norm)
        
        # pooling layers
        self.GP_module = GlobalSAModule(nhidden, nhidden, dropout=dropout, pool_type=pool_type)
        self.SA_modules = torch.nn.ModuleList()
        self.FP_modules = torch.nn.ModuleList()
        for i in range(depth):
            self.SA_modules.append(
                SAModule(
                    nhidden,
                    nhidden,
                    ratios[i],
                    radii[i], 
                    max_neighbors=max_neighbors,
                    aggr=aggr,
                    batch_norm=batch_norm
                )
            )
            self.FP_modules.append(
                FPModule(
                    nhidden,
                    nhidden,
                    nhidden,
                    k=knn_num,
                    batch_norm=batch_norm
                )
            )
        
        # linear layers
        self.lin_out1 = MLP(
            [nhidden, nhidden, nOut1],
            batch_norm=[batch_norm, False],
            act=['relu', None]
        )
        self.lin_out2 = MLP(
            [nhidden, nhidden, nOut2],
            batch_norm=[batch_norm, False],
            act=['relu', None]
        )
        self.nout = (nOut1, nOut2)
    
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
        
        # segmentation
        # knn unpooling
        fp_out = self.FP_modules[-1].forward(*sa_outs[-1], *sa_outs[-2])
        for i in range(1, self.depth):
            j = - i
            fp_out = self.FP_modules[j-1].forward(*fp_out, *sa_outs[j-2])
        x1 = fp_out[0]
        
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.lin_out1(x1)
        
        # global pooling
        x2 = self.GP_module(*sa_outs[-1])
        
        # lin2
        x2 = F.dropout(x2, self.dropout, self.training)
        x2 = self.lin_out2(x2)
        
        return x1, x2
