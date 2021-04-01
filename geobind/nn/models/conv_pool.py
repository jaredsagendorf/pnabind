# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GMMConv
from torch_geometric.transforms import Compose, PointPairFeatures, GenerateMeshNormals
from torch_geometric.nn import PairNorm, InstanceNorm, BatchNorm
from torch_geometric.utils import dropout_adj

# geobind modules
from geobind.nn.layers import EdgePooling, MeshPooling
from geobind.nn.layers import ContinuousCRF
from geobind.nn.transforms import GeometricEdgeFeatures, ScaleEdgeFeatures
from geobind.nn.utils import MLP

class NetConvPool(torch.nn.Module):
    def __init__(self, nIn, nOut=None,
            conv_args={},
            pool_args={},
            crf_args={},
            nhidden=16,
            nhidden_lin=16,
            level_hidden_size_down=None,
            level_hidden_size_up=None,
            depth=2,
            edge_dim=4,
            #edge_components=6,
            transform_args=None,
            num_top_convs=3,
            num_pool_convs=1,
            num_unpool_convs=1,
            act='relu',
            name='net_conv_pool',
            use_lin=True,
            use_crf=False,
            scale_edge_features=None,
            top_conv_aggr='mean',
            down_conv_aggr='mean',
            up_conv_aggr='mean',
            norm='MLP_only',
            norm_kwargs={},
            v_dropout=0.0,
            e_dropout=0.0,
        ):
        super(NetConvPool, self).__init__()
        self.depth = depth
        self.num_top_convs = num_top_convs
        self.num_pool_convs = num_pool_convs
        self.num_unpool_convs = num_unpool_convs
        self.name = name
        self.use_lin = use_lin
        self.use_crf = use_crf
        self.e_dropout = e_dropout
        self.v_dropout = v_dropout
        
        # get activation function
        if act == 'relu':
            self.act = F.relu
        elif act == 'elu':
            self.act = F.elu
        elif act == 'selu':
            self.act = F.selu
        else:
            self.act = act # assume we are passed a function
        
        # determine pooling types
        # if depth > 0:
            # self.pool_type = pool_args["name"]
            
            # # build transforms
            # if self.pool_type == "MeshPool":
                # transforms = [GeometricEdgeFeatures(n_components=edge_components)]
                # self.edge_dim = transforms[0].edge_dim
            # elif self.pool_type == "EdgePool":
                # self.edge_dim = 4
                # transforms = [GenerateMeshNormals(), PointPairFeatures()]
            # else:
                # raise ValueError("Unknown pooling type: {}".format(self.pool_type))
        # else:
            # transforms = []
            # self.edge_dim = edge_dim
        
        # if scale_edge_features:
            # transforms.append(ScaleEdgeFeatures(method=scale_edge_features))
        # self.transforms = Compose(transforms)
        self.edge_dim = edge_dim
        
        # containers to hold the layers
        self.top_convs = torch.nn.ModuleList()
        if self.depth > 0:
            self.down_pools = torch.nn.ModuleList()
            self.down_convs = torch.nn.ModuleList()
            self.up_convs = torch.nn.ModuleList()
        if norm in ('PairNorm', 'InstanceNorm', 'BatchNorm'):
            self.norm_layers = torch.nn.ModuleList()
        else:
            self.norm_layers = None
        self.norm = norm
        
        # set hidden size of every level    
        if level_hidden_size_down is None:
            level_hidden_size_down = [nhidden]*(depth+1)
        assert len(level_hidden_size_down) == depth + 1
        
        if depth > 0:
            if level_hidden_size_up is None:
                level_hidden_size_up = level_hidden_size_down[(depth-1)::-1]
            level_hidden_size_up = [level_hidden_size_down[-1]] + level_hidden_size_up
            assert len(level_hidden_size_up) == depth + 1
        
        # down FC layer (dimentionality reduction)
        bn_norm = (norm is not None)
        self.lin_in = MLP([nIn, nIn, nhidden_lin], batch_norm=bn_norm)
        
        # top convolutions
        ks = conv_args.get("kernel_size", 6)
        nin = nhidden_lin
        nout = level_hidden_size_down[0]
        for i in range(num_top_convs):
            self.top_convs.append(GMMConv(nin, nout, self.edge_dim, ks, aggr=top_conv_aggr))
            nin = nout
        if norm:
            self.addNormalizationLayer(nout, norm, norm_kwargs)
        
        # down layers
        nin = nout
        for i in range(depth):
            nout = level_hidden_size_down[i+1]
            # pooling
            self.down_pools.append(self.makePool(nin, **pool_args))
            
            # convolution
            for j in range(num_pool_convs):
                self.down_convs.append(GMMConv(nin, nout, self.edge_dim, ks, aggr=down_conv_aggr))
                nin = nout
            if norm:
                self.addNormalizationLayer(nout, norm, norm_kwargs)
        
        # up layers
        for i in range(depth):
            j = depth - i - 1
            nin = level_hidden_size_up[i] + level_hidden_size_down[j]# if use_skips else level_hidden_size_up[i]
            nout = level_hidden_size_up[i+1]
            
            # convolution
            for k in range(num_unpool_convs):
                self.up_convs.append(GMMConv(nin, nout, self.edge_dim, ks, aggr=up_conv_aggr))
                nin = nout
            if norm:
                self.addNormalizationLayer(nout, norm, norm_kwargs)
        self.nout = nout
        
        if use_lin:
            ## final FC layers
            self.nout = nOut
            self.lin_out = MLP([nout, nout, nOut], batch_norm=False, act=['relu', None])
        
        if use_crf:
            # continuous CRF layer
            self.crf1 = ContinuousCRF(**crf_args)
    
    def makePool(self, nin, name=None, **kwargs):
        if name == "EdgePool":
            # Edge Pooling
            return EdgePooling(nin, self.edge_dim, post_transform=self.transforms, **kwargs)
        
        if name == "MeshPool":
            # Mesh pooling
            return MeshPooling(nin, self.edge_dim, post_transform=self.transforms, **kwargs)
    
    def addNormalizationLayer(self, channels, norm, norm_kwargs):
        if norm is None:
            pass
        elif norm == "MLP_only":
            pass
        elif norm == 'PairNorm':
            self.norm_layers.append(PairNorm(**norm_kwargs))
        elif norm == 'InstanceNorm':
            self.norm_layers.append(InstanceNorm(channels, **norm_kwargs))
        elif norm == 'BatchNorm':
            self.norm_layers.append(BatchNorm(channels, **norm_kwargs))
        else:
            raise ValueError("unrecognized normalization type: {}".format(norm))
    
    def forward(self, data):
        skip_connections = []
        graph_activations = []
        if self.depth > 0:
            graph_activations.append(data)
        
        # lin1
        x = self.lin_in(data.x)
        x = F.dropout(x, p=self.v_dropout, training=self.training)
        
        # top convs
        edge_index, edge_attr = dropout_adj(data.edge_index, # edge dropout
            edge_attr=data.edge_attr,
            p=self.e_dropout,
            training=self.training,
            num_nodes=x.size(0)
        )
        for i in range(self.num_top_convs):
            x = self.act(self.top_convs[i](x, edge_index, edge_attr))
        x = F.dropout(x, p=self.v_dropout, training=self.training)
        
        if self.norm_layers is not None:
            # normalization
            if self.norm == "BatchNorm":
                x = self.norm_layers[0].forward(x)
            else:
                x = self.norm_layers[0].forward(x, batch=data.batch)
        
        # store skip-connection
        skip_connections.append(x)
        
        # down conv-pools
        for i in range(self.depth):
            # pooling 
            x, data = self.down_pools[i](x, data) # data stores new pos, faces, edge_index, edge_attr, batch and unpooling info
            graph_activations.append(data)
            
            # convolution
            edge_index, edge_attr = dropout_adj(data.edge_index, # edge dropout
                edge_attr=data.edge_attr,
                p=self.e_dropout,
                training=self.training,
                num_nodes=x.size(0)
            )
            for j in range(self.num_pool_convs):
                x = self.act(self.down_convs[i*self.num_pool_convs + j](x, edge_index, edge_attr))
            x = F.dropout(x, p=self.v_dropout, training=self.training) # vertex dropout
            
            if self.norm_layers is not None:
                # normalization
                if self.norm == "BatchNorm":
                    x = self.norm_layers[i+1].forward(x)
                else:
                    x = self.norm_layers[i+1].forward(x, batch=data.batch)
            
            # store skip-connection
            if (i+1) < self.depth:
                skip_connections.append(x)
        
        # up pool-convs
        for i in range(self.depth):
            j = self.depth - 1 - i
            
            # unpool
            x, edge_index, batch = self.down_pools[j].unpool(x, graph_activations[j+1].unpool_info)
            
            # convolution
            x = torch.cat([x, skip_connections[j]], axis=-1)
            data = graph_activations[j]
            
            edge_index, edge_attr = dropout_adj(data.edge_index, # edge dropout
                edge_attr=data.edge_attr,
                p=self.e_dropout,
                training=self.training,
                num_nodes=x.size(0)
            )
            for j in range(self.num_unpool_convs):
                x = self.act(self.up_convs[i*self.num_unpool_convs + j](x, edge_index, edge_attr))
            x = F.dropout(x, p=self.v_dropout, training=self.training) # vertex dropout
            
            if self.norm_layers is not None:
                # normalization
                if self.norm == "BatchNorm":
                    x = self.norm_layers[self.depth+i+1].forward(x)
                else:
                    x = self.norm_layers[self.depth+i+1].forward(x, batch=data.batch)
        
        # lin 2-4
        if self.use_lin:
            #x = self.act(self.lin2(x))
            #x = self.act(self.lin3(x))
            #x = self.lin4(x)
            x = self.lin_out(x)
        
        # crf layer
        if self.use_crf:
            x = self.crf1(x, data.edge_index)
        
        return x
