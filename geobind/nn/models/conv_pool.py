# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GMMConv
from torch_geometric.transforms import Compose, PointPairFeatures, GenerateMeshNormals

# geobind modules
from geobind.nn.layers import EdgePooling, MeshPooling
from geobind.nn.layers import ContinuousCRF
from geobind.nn.transforms import GeometricEdgeFeatures, ScaleEdgeFeatures

class NetConvPool(torch.nn.Module):
    def __init__(self, nIn, nOut=None,
            conv_args={},
            pool_args={},
            crf_args={},
            nhidden=32,
            nhidden_lin=16,
            level_hidden_size_down=None,
            level_hidden_size_up=None,
            depth=2,
            num_top_convs=3,
            num_pool_convs=1,
            num_unpool_convs=1,
            act='relu',
            use_skips=True,
            #sum_skips=False,
            dropout=0.5,
            name='net_conv_pool',
            use_lin=True,
            use_crf=False,
            scale_edge_features=None
        ):
        super(NetConvPool, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.use_skips = use_skips
        #self.sum_skips = sum_skips
        self.num_top_convs = num_top_convs
        self.num_pool_convs = num_pool_convs
        self.num_unpool_convs = num_unpool_convs
        self.name = name
        self.lin = use_lin
        self.crf = use_crf
        
        # get activation function
        if(act == 'relu'):
            self.act = F.relu
        elif(act == 'elu'):
            self.act = F.elu
        elif(act == 'selu'):
            self.act = F.selu
        else:
            self.act = act # assume we are passed a function
        
        # determine pooling types
        if depth > 0:
            self.pool_type = pool_args["name"]
        
        # build transforms
        if self.pool_type == "MeshPool":
            self.edge_dim = 9
            transforms = [GeometricEdgeFeatures()]
        elif self.pool_type == "EdgePool":
            self.edge_dim = 4
            transforms = [GenerateMeshNormals(), PointPairFeatures()]
        else:
            raise ValueError("Unknown pooling type: {}".format(self.pool_type))
        
        if scale_edge_features:
            transforms.append(ScaleEdgeFeatures(method=scale_edge_features))
        self.transforms = Compose(transforms)
        
        # containers to hold the layers
        self.top_convs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        
        # set hidden size of every level    
        if level_hidden_size_down is None:
            level_hidden_size_down = [nhidden]*(depth+1)
        
        if level_hidden_size_up is None:
            level_hidden_size_up = level_hidden_size_down[(depth-1)::-1]
        level_hidden_size_up = [level_hidden_size_down[-1]] + level_hidden_size_up

        assert len(level_hidden_size_down) == depth + 1
        assert len(level_hidden_size_up) == depth + 1
        
        # down FC layer (dimentionality reduction)
        self.lin1 = nn.Linear(nIn, nhidden_lin)
        
        # top convolutions
        nin = nhidden_lin
        nout = level_hidden_size_down[0]
        for i in range(num_top_convs):
            self.top_convs.append(GMMConv(nin, nout, self.edge_dim, conv_args["kernel_size"]))
            nin = nout
        
        # down layers
        nin = nout
        for i in range(depth):
            nout = level_hidden_size_down[i+1]
            # pooling
            self.down_pools.append(self.makePool(nin, **pool_args))
            
            # convolution
            for j in range(num_pool_convs):
                self.down_convs.append(GMMConv(nin, nout, self.edge_dim, conv_args["kernel_size"]))
                nin = nout
        
        # up layers
        for i in range(depth):
            j = depth - i - 1
            nin = level_hidden_size_up[i] + level_hidden_size_down[j] if use_skips else level_hidden_size_up[i]
            nout = level_hidden_size_up[i+1]
            
            # convolution
            for k in range(num_unpool_convs):
                self.up_convs.append(GMMConv(nin, nout, self.edge_dim, conv_args["kernel_size"]))
                nin = nout
        self.nout = nout
        
        if use_crf:
            # continuous CRF layer
            self.crf1 = ContinuousCRF(**crf_args)
        
        if use_lin:
            # final FC layers
            self.lin2 = nn.Linear(nout, nout)
            self.lin3 = nn.Linear(nout, nout)
            self.lin4 = nn.Linear(nout, nOut)
            self.nout = nOut
    
    def makePool(self, nin, name=None, **kwargs):
        if name == "EdgePool":
            # Edge Pooling
            return EdgePooling(nin, self.edge_dim, post_transform=self.transforms, **kwargs)
        
        if name == "MeshPool":
            # Mesh pooling
            return MeshPooling(nin, self.edge_dim, post_transform=self.transforms, check_manifold=True, **kwargs)
    
    def forward(self, data):
        skip_connections = []
        graph_activations = []
        if self.depth > 0:
            graph_activations.append(data)
        
        # lin1
        x = self.act(self.lin1(data.x))
        
        # top convs
        for i in range(self.num_top_convs):
            x = self.act(self.top_convs[i](x, data.edge_index, data.edge_attr))
        if self.use_skips:
            skip_connections.append(x)
        
        # down conv-pools
        for i in range(self.depth):
            # pooling 
            x, data = self.down_pools[i](x, data) # data stores new pos, faces, edge_index, edge_attr, batch and unpooling info
            graph_activations.append(data)
            
            # convolution
            for j in range(self.num_pool_convs):
                x = self.act(self.down_convs[i*self.num_pool_convs + j](x, data.edge_index, data.edge_attr))
            if self.use_skips and (i+1) < self.depth:
                skip_connections.append(x)
        
        # up pool-convs
        for i in range(self.depth):
            j = self.depth - 1 - i
            
            # unpool
            x, edge_index, batch = self.down_pools[j].unpool(x, graph_activations[j+1].unpool_info)
            
            # convolution
            if self.use_skips:
                x = torch.cat([x, skip_connections[j]], axis=-1)
            data = graph_activations[j]
            
            for j in range(self.num_unpool_convs):
                x = self.act(self.up_convs[i*self.num_unpool_convs + j](x, data.edge_index, data.edge_attr))
        
        # crf layer
        if self.crf:
            x = self.crf1(x, data.edge_index)
        
        # lin 2-4
        if self.lin:
            x = self.act(self.lin2(x))
            x = self.act(self.lin3(x))
            x = self.lin4(x)
        
        return x
