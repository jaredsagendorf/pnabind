# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius, fps, knn_interpolate
from torch_geometric.nn import PointConv, PPFConv, GraphConv
from torch_geometric.nn import PairNorm, InstanceNorm, BatchNorm
from torch_geometric.utils import dropout_adj

# geobind modules
from geobind.nn.layers import ContinuousCRF
from geobind.nn.utils import MLP

class SAModule(torch.nn.Module):
    def __init__(self, nIn, nOut, conv_args, ratio, radius,
            max_neighbors=32,
            batch_norm=False,
            graph_norm=None,
            graph_norm_kwargs={},
            e_dropout=0.0,
            v_dropout=0.0
        ):
        super(SAModule, self).__init__()
        """This module acts as a pooling/conv layer. Taken from pytorch-geometric examples."""
        self.ratio = ratio
        self.r = radius
        self.K = max_neighbors
        self.e_dropout = e_dropout
        self.v_dropout = v_dropout
        
        # set up normalization
        if graph_norm == 'PairNorm':
            self.norm = PairNorm(**graph_norm_kwargs)
        elif graph_norm == 'InstanceNorm':
            self.norm = InstanceNorm(nOut, **graph_norm_kwargs)
        elif graph_norm == 'BatchNorm':
            self.norm = BatchNorm(nOut, **graph_norm_kwargs)
        else:
            self.norm = None
        
        # set up convolution
        self.conv_name = conv_args['name']
        if conv_args['name'] == 'PointConv':
            dim = nIn + 3
            nn = MLP([dim, dim, nOut], batch_norm=batch_norm)
            self.conv = PointConv(nn)
        elif conv_args['name'] == 'GraphConv':
            self.conv =  GraphConv(nIn, nOut, aggr=conv_args['aggr'])
        elif conv_args['name'] == 'PPFConv':
            dim = nIn + 4
            nn1 = MLP([dim, dim, dim], batch_norm=batch_norm)
            nn2 = MLP([dim, nOut], batch_norm=batch_norm)
            self.conv = PPFConv(nn1, nn2, add_self_loops=False)
            self.conv.aggr = conv_args.get('aggr', 'max')
        
    def forward(self, x, pos, batch, norm=None):
        # pool points based on FPS algorithm, returning Npt*ratio centroids
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        
        # finds points within radius `self.r` of the centroids, up to `self.K` pts per centroid
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=self.K)
        
        # perform convolution over edges joining centroids to their neighbors within ball of radius `self.r`
        num_nodes = x.size(0)
        if self.conv_name == 'PointConv':
            edge_index, _ = dropout_adj(
                torch.stack([col, row], dim=0),
                p=self.e_dropout,
                training=self.training,
                num_nodes=num_nodes
            )
            x = self.conv(x, (pos, pos[idx]), edge_index)
        elif self.conv_name == 'GraphConv':
            row = idx[row]
            edge_index, _ = dropout_adj(
                torch.stack([col, row], dim=0),
                p=self.e_dropout,
                training=self.training,
                num_nodes=num_nodes
            )
            x = self.conv(x, edge_index)[idx]
        elif self.conv_name == 'PPFConv':
            row = idx[row]
            edge_index, _ = dropout_adj(
                torch.stack([col, row], dim=0),
                p=self.e_dropout,
                training=self.training,
                num_nodes=num_nodes
            )
            x = self.conv(x, pos, norm, edge_index)[idx]
        pos, batch = pos[idx], batch[idx]
        
        # perform normalization
        if self.norm is not None:
            if isinstance(self.norm, BatchNorm):
                x = self.norm(x)
            else:
                x = self.norm(x, batch=batch)
        
        # perform vertex dropout
        x = F.dropout(x, p=self.v_dropout, training=self.training)
        
        return x, pos, batch, idx

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

class PointNetPP(torch.nn.Module):
    def __init__(self, nIn, nOut=None, 
            conv_args=None,
            crf_kwargs={},
            depth=3,
            nhidden=16,
            ratios=None,
            radii=None,
            max_neighbors=32,
            knn_num=3,
            name='pointnet_pp',
            use_lin=True,
            use_crf=False,
            batch_norm=True,
            graph_norm=False,
            graph_norm_kwargs={},
            v_dropout=0.0,
            e_dropout=0.0
        ):
        super(PointNetPP, self).__init__()
        
        ### Set up ###
        self.depth = depth
        self.name = name
        self.conv_name = conv_args["name"]
        self.use_lin = use_lin
        self.use_crf = use_crf
        self.v_dropout = v_dropout
        self.e_dropout = e_dropout
        
        if ratios is None:
            ratios = [0.5]*depth
        assert len(ratios) == depth
        
        if radii is None:
            radii = [2.0]*depth
        assert len(radii) == depth
        
        ### Model Layers ###
        # linear input
        self.lin_in = MLP([nIn, nhidden, nhidden])
        
        # pooling/unpooling layers
        self.SA_modules = torch.nn.ModuleList()
        self.FP_modules = torch.nn.ModuleList()
        for i in range(depth):
            self.SA_modules.append(
                SAModule(
                    nhidden,
                    nhidden,
                    conv_args,
                    ratios[i],
                    radii[i], 
                    max_neighbors=max_neighbors,
                    e_dropout=e_dropout,
                    v_dropout=v_dropout,
                    batch_norm=batch_norm,
                    graph_norm=graph_norm,
                    graph_norm_kwargs=graph_norm_kwargs
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
        self.nout = nhidden
        
        # linear layers
        if use_lin:
            self.lin_out = MLP(
                    [nhidden, nhidden, nOut],
                    batch_norm=[batch_norm, False],
                    act=['relu', None],
                    dropout=[dropout, 0.0]
            )
            self.nout = nOut
        
        if use_crf:
            self.crf1 = ContinuousCRF(**crf_kwargs)
    
    def forward(self, data):
        # lin1
        x = self.lin_in(data.x)
        x = F.dropout(x, p=self.v_dropout, training=self.training)
        
        # conv/pooling
        norm = data.norm
        sa_outs = [(x, data.pos, data.batch)]
        for i in range(self.depth):
            if self.conv_name == 'PPFConv':
                args = (*sa_outs[i], norm)
            else:
                args = sa_outs[i]
            x, pos, batch, idx = self.SA_modules[i].forward(*args)
            sa_outs.append((x, pos, batch))
            norm = norm[idx]
        
        # unpooling
        fp_out = self.FP_modules[-1].forward(*sa_outs[-1], *sa_outs[-2])
        for i in range(1, self.depth):
            j = - i
            fp_out = self.FP_modules[j-1].forward(*fp_out, *sa_outs[j-2])
        x = fp_out[0]
        
        # lin 2-4
        if self.use_lin:
            x = F.dropout(x, p=self.v_dropout, training=self.training)
            x = self.lin_out(x)
        
        # crf layer
        if self.use_crf:
            x = self.crf1(x, data.edge_index)
        
        return x
