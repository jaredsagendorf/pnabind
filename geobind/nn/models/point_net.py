# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, Sequential as Seq, Linear as Lin, BatchNorm1d as BN
from torch_geometric.nn import radius, fps, knn_interpolate
from torch_geometric.nn import PointConv, PPFConv, GraphConv
from torch_geometric.nn import PairNorm, InstanceNorm

# geobind modules
from geobind.nn.layers import ContinuousCRF

def MLP(channels, batch_norm=True):
    if batch_norm:
        return Seq(*[
                Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
                for i in range(1, len(channels))
            ])
    else:
        return Seq(*[
                Seq(Lin(channels[i - 1], channels[i]), ReLU())
                for i in range(1, len(channels))
            ])

class SAModule(torch.nn.Module):
    def __init__(self, nIn, nOut, conv_args, ratio, radius,
            max_neighbors=64,
            norm=None,
            norm_kwargs={}
        ):
        super(SAModule, self).__init__()
        """This module acts as a pooling/conv layer. Taken from pytorch-geometric examples."""
        self.ratio = ratio
        self.r = radius
        self.K = max_neighbors
        
        # set up normalization
        if norm == 'PairNorm':
            self.norm = PairNorm(**norm_kwargs)
            bn = False
        elif norm == 'InstanceNorm':
            self.norm = InstanceNorm(nOut, **norm_kwargs)
            bn = False
        else:
            self.norm = None
            bn = True
        
        # set up convolution
        self.conv_name = conv_args['name']
        if conv_args['name'] == 'PointConv':
            dim = nIn + 3
            nn = MLP([dim, dim, nOut], batch_norm=bn)
            self.conv = PointConv(nn)
        elif conv_args['name'] == 'GraphConv':
            self.conv =  GraphConv(nIn, nOut, aggr=conv_args['aggr'])
        elif conv_args['name'] == 'PPFConv':
            dim = nIn + 4
            nn = MLP([dim, dim, nOut], batch_norm=bn)
            self.conv = PPFConv(nn)
            self.conv.aggr = conv_args['aggr']
        
    def forward(self, x, pos, batch, norm=None):
        # pool points based on FPS algorithm, returning Npt*ratio centroids
        idx = fps(pos, batch, ratio=self.ratio) 
        
        # finds points within radius `self.r` of the centroids, up to `self.K` pts per centroid
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=self.K)
        
        # perform convolution over edges joining centroids to their neighbors within ball of radius `self.r`
        if self.conv_name == 'PointConv':
            edge_index = torch.stack([col, row], dim=0)
            x = self.conv(x, (pos, pos[idx]), edge_index)
        elif self.conv_name == 'GraphConv':
            row = idx[row]
            edge_index = torch.stack([col, row], dim=0)
            x = self.conv(x, edge_index)[idx]
        elif self.conv_name == 'PPFConv':
            row = idx[row]
            edge_index = torch.stack([col, row], dim=0)
            x = self.conv(x, pos, norm, edge_index)[idx]
        pos, batch = pos[idx], batch[idx]
        
        # perform normalization
        if self.norm is not None:
            x = self.norm(x, batch=batch)
        
        return x, pos, batch, idx

class FPModule(torch.nn.Module):
    def __init__(self, nIn, nSkip, nOut, k=3):
        super(FPModule, self).__init__()
        """This module acts as an unpooling/interpolation layer. Taken from pytorch-geometric examples."""
        self.k = k
        dim = nIn + nSkip
        self.nn = MLP([dim, dim, nOut], batch_norm=True)
    
    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        
        return x, pos_skip, batch_skip

class PointNetPP(torch.nn.Module):
    def __init__(self, nIn, nOut=None, 
            conv_args=None,
            crf_args={},
            depth=3,
            nhidden=32,
            ratios=None,
            radii=None,
            max_neighbors=64,
            knn_num=3,
            name='pointnet_pp',
            use_lin=True,
            use_crf=False,
            norm=None,
            norm_kwargs={},
            dropout=0
        ):
        super(PointNetPP, self).__init__()
        ### Set up ###
        self.depth = depth
        self.name = name
        self.conv_name = conv_args["name"]
        self.use_lin = use_lin
        self.use_crf = use_crf
        self.dropout = dropout
        
        if ratios is None:
            ratios = [0.5]*depth
        assert len(ratios) == depth
        
        if radii is None:
            radii = [2.0]*depth
        assert len(radii) == depth
        
        # activation function
        #self.act = ReLU()
        #if(act == 'relu'):
            #self.act = F.relu
        #elif(act == 'elu'):
            #self.act = F.elu
        #elif(act == 'selu'):
            #self.act = F.selu
        
        ### Model Layers ###
        # dimentionality reduction
        self.lin_in = nn.Linear(nIn, nhidden)
        
        # pooling/unpooling layers
        self.SA_modules = torch.nn.ModuleList()
        self.FP_modules = torch.nn.ModuleList()
        for i in range(depth):
            self.SA_modules.append(SAModule(nhidden, nhidden, conv_args, ratios[i], radii[i], max_neighbors=max_neighbors, norm=norm, norm_kwargs=norm_kwargs))
            self.FP_modules.append(FPModule(nhidden, nhidden, nhidden, k=knn_num))
        self.nout = nhidden
        
        if use_crf:
            self.crf1 = ContinuousCRF(**crf_args)
            
        # linear layers
        if use_lin:
            self.lin_out = MLP([nhidden, nhidden, nOut], batch_norm=True)
            #self.lin2 = torch.nn.Linear(nhidden, nhidden)
            #self.lin3 = torch.nn.Linear(nhidden, nhidden)
            #self.lin4 = torch.nn.Linear(nhidden, nOut)
            self.nout = nOut
    
    def forward(self, data):
        # lin1
        x = F.relu(self.lin_in(data.x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # pooling
        norm = data.norm
        sa_outs = [(x, data.pos, data.batch)]
        for i in range(self.depth):
            if self.conv_name == 'PPFConv':
                args = (*sa_outs[i], norm)
            else:
                args = sa_outs[i]
            x, pos, batch, idx = self.SA_modules[i].forward(*args)
            x = F.dropout(x, p=self.dropout, training=self.training)
            sa_outs.append((x, pos, batch))
            norm = norm[idx]
        
        # unpooling
        fp_out = self.FP_modules[-1].forward(*sa_outs[-1], *sa_outs[-2])
        for i in range(1, self.depth):
            j = - i
            fp_out = self.FP_modules[j-1].forward(*fp_out, *sa_outs[j-2])
        x = fp_out[0]
        
        # crf layer
        if self.use_crf:
            x = self.crf1(x, data.edge_index)
            
        # lin 2-4
        if self.use_lin:
            x = F.dropout(x, p=self.dropout, training=self.training)
            #x = self.act(self.lin2(x))
            #x = self.act(self.lin3(x))
            #x = self.lin4(x)
            x = self.lin_out(x)
        
        return x
