# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, GMMConv, NNConv, CGConv, PPFConv, FeaStConv 
from torch_geometric.transforms import Compose, PointPairFeatures, GenerateMeshNormals
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh

# geobind modules
from geobind.nn.layers import EdgePooling, MeshPooling
from geobind.nn.layers import ContinuousCRF
from geobind.nn.transforms import GeometricEdgeFeatures, ScaleEdgeFeatures

class NetConvEdgePool(torch.nn.Module):
    def __init__(self, nIn, nOut=None,
            conv_args={},
            pool_args={},
            crf_args={},
            nhidden=32,
            depth=2,
            num_top_convs=0,
            num_bottom_convs=1,
            act='relu',
            use_skips=True,
            sum_skips=False,
            dropout=0.5,
            edge_dim=9,
            name='net_conv_edge_pool',
            use_lin=True,
            use_crf=False,
            scale_edge_features=None
        ):
        super(NetConvEdgePool, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.use_skips = use_skips
        self.sum_skips = sum_skips
        self.num_top_convs = num_top_convs
        self.num_bottom_convs = num_bottom_convs*(depth > 0)
        self.edge_dim = edge_dim
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
        
        # determine convolution and pooling types
        self.conv_type = conv_args["name"]
        if depth > 0:
            self.pool_type = pool_args["name"]
        
        # build transforms
        transforms = [GeometricEdgeFeatures()]
        if scale_edge_features:
            transforms.append(ScaleEdgeFeatures(method=scale_edge_features))
        self.transforms = Compose(transforms)
        
        # containers to hold the layers
        self.top_convs = torch.nn.ModuleList()
        self.down_convs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.bottom_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        
        # set size of every layer
        num_lin = 4 if use_lin else 1 # 1 + 3
        if isinstance(nhidden, int):
            nhidden = [nhidden]*(self.num_top_convs + self.depth + self.num_bottom_convs + self.depth + num_lin)
        assert len(nhidden) ==  (self.num_top_convs + self.depth + self.num_bottom_convs + self.depth + num_lin)
        
        # down FC layer (dimentionality reduction)
        self.lin1 = nn.Linear(nIn, nhidden[0])
        
        # top convolutions
        for i in range(num_top_convs):
            self.top_convs.append(self.makeConv(nhidden[i], nhidden[i+1], conv_args))
        
        # down layers
        channels = nhidden
        for i in range(num_top_convs, num_top_convs + depth):
            # convolution
            self.down_convs.append(self.makeConv(nhidden[i], nhidden[i+1], conv_args))
            
            # pooling
            self.down_pools.append(self.makePool(nhidden[i+1], **pool_args))
        
        # bottom convolutions
        if depth > 0:
            for i in range(num_top_convs + depth, num_top_convs + depth + num_bottom_convs):
                self.bottom_convs.append(self.makeConv(nhidden[i], nhidden[i+1], conv_args))
        
        # up layers
        if depth > 0:
            offset = 2*num_top_convs + 2*depth + num_bottom_convs
            for i in range(num_top_convs + depth + num_bottom_convs, num_top_convs + depth + num_bottom_convs + depth):
                if self.use_skips:
                    j = offset - i
                    if sum_skips and nhidden[i] == nhidden[j]:
                        in_channels = nhidden[i]
                        out_channels = nhidden[i+1]
                    else:
                        in_channels = nhidden[i] + nhidden[j]
                        out_channels = nhidden[i+1]
                else:
                    in_channels = nhidden[i]
                    out_channels = nhidden[i+1]
                self.up_convs.append(self.makeConv(in_channels, out_channels, conv_args))
        
        if use_crf:
            self.crf1 = ContinuousCRF(**crf_args)
        
        if use_lin:
            # up FC layers
            self.lin2 = nn.Linear(nhidden[-3], nhidden[-2])
            self.lin3 = nn.Linear(nhidden[-2], nhidden[-1])
            self.lin4 = nn.Linear(nhidden[-1], nOut)
    
    def makePool(self, nin, name=None, **kwargs):
        # Edge Pool
        if name == "EdgePool":
            return EdgePooling(nin, self.edge_dim, **kwargs)
        # Edge Pool with edge features
        if name == "MeshPool":
            return MeshPooling(nin, self.edge_dim, post_transform=self.transforms, check_manifold=True, **kwargs)
    
    def makeConv(self, nin, nout, conv_args):
        # FeaStConv
        if(conv_args['name'] == 'FeaSt'):
            return FeaStConv(nin, nout, conv_args["num_heads"])
        
        # SplineConv
        if(conv_args['name'] == 'Spline'):
            return SplineConv(nin, nout, 
                self.edge_dim,
                conv_args["kernel_size"],
                is_open_spline=conv_args["open_spline"],
                degree=conv_args["degree"]
            )
        
        # GMMConv
        if(conv_args['name'] == "GMM"):
            return GMMConv(nin, nout,
                self.edge_dim,
                conv_args["kernel_size"]
            )
        
        # NNConv
        if(conv_args["name"] == "NN"):
            h = nn.Sequential(
                    nn.Linear(self.edge_dim, nin*nout),
                    nn.ReLU()
                    #nn.Linear(int(nin*nout/2), nin*nout)
            )
            return NNConv(nin, nout, h)
        
        # PPFConv
        if(conv_args["name"] == "PPF"):
            cin = nin+4
            hl = nn.Sequential(
                nn.Linear(cin, conv_args['nhidden']),
                nn.ReLU()
            )
            hg = nn.Linear(conv_args['nhidden'], nout)
            #hl = nn.Sequential(
                    #nn.Linear(cin, int(conv_args['nhidden']/2)),
                    #nn.ReLU(),
                    #nn.Linear(int(conv_args['nhidden']/2), conv_args['nhidden'])
            #)
            #hg = nn.Sequential(
                    #nn.Linear(conv_args['nhidden'], nout),
                    #nn.ReLU(),
                    #nn.Linear(nout, nout)
            #)
            return PPFConv(hl, hg)
        
        # CGConv
        if(conv_args["name"] == "CG"):
            return CGConv(nin, self.edge_dim)
    
    def getPoolArgs(self, x, data):
        if self.pool_type == 'EdgePool':
            return (x, data.edge_index, data.batch)
        elif self.pool_type == 'MeshPool':
            return (x, data)
    
    def getConvArgs(self, x, data):
        if(self.conv_type == "FeaSt"):
            return (x, data.edge_index)
        if(self.conv_type == "Spline"):
            return (x, data.edge_index, data.edge_attr)
        if(self.conv_type == "GMM"):
            return (x, data.edge_index, data.edge_attr)
        if(self.conv_type == "NN"):
            return (x, data.edge_index, data.edge_attr)
        if(self.conv_type == "PPF"):
            return (x, data.pos, data.norm, data.edge_index)
        if(self.conv_type == "CG"):
            return (x, data.edge_index, data.edge_attr)
    
    def forward(self, data):
        skip_connections = []
        graph_activations = []
        if self.depth > 0:
            graph_activations.append(data)
        
        # lin1
        x = self.act(self.lin1(data.x))
        
        # top convs
        for i in range(self.num_top_convs):
            args = self.getConvArgs(x, data)
            x = self.act(self.top_convs[i](*args))
        
        # down conv-pools
        for i in range(self.depth):
            # convolution
            args = self.getConvArgs(x, graph_activations[i])
            x = self.act(self.down_convs[i](*args))
            if self.use_skips:
                skip_connections.append(x)
            
            # pooling 
            args = self.getPoolArgs(x, graph_activations[i])
            data_pooled = self.down_pools[i](*args)
            graph_activations.append(data_pooled)
            x = data_pooled.x
        
        # bottom convs
        for i in range(self.num_bottom_convs):
            args = self.getConvArgs(x, graph_activations[-1])
            x = self.act(self.bottom_convs[i](*args))
        
        # up pool-convs
        for i in range(self.depth):
            j = self.depth - 1 - i
            
            # unpool
            x, edge_index, batch = self.down_pools[j].unpool(x, graph_activations[j+1].unpool_info)
            
            # convolution
            if(self.use_skips):
                x = x + skip_connections[j] if self.sum_skips else torch.cat([x, skip_connections[j]], axis=-1)
            args = self.getConvArgs(x, graph_activations[j])
            
            x = self.act(self.up_convs[i](*args))
        
        # crf layer
        if self.crf:
            x = self.crf1(x, args[1])
        
        # lin 2-4
        if self.lin:
            x = self.act(self.lin2(x))
            x = self.act(self.lin3(x))
            x = self.lin4(x)
        
        return x
    
    def predict(self, data, threshold=0.5):
        with torch.no_grad():
            x = self.forward(data)
            prob = F.softmax(x, dim=1)
            
            return prob, (pred[:,1] >= threshold).long()
