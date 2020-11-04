# third party packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  FeaStConv, SplineConv, EdgePooling, GMMConv, NNConv, PPFConv, CGConv
from torch_geometric.transforms import PointPairFeatures, GenerateMeshNormals
from torch_geometric.data import Data

# geobind packages
from geobind.nn.layers import EdgePooling_EF
from geobind.nn.layers import ContinuousCRF

class NetConvEdgePool(torch.nn.Module):
    GMN = GenerateMeshNormals()
    PPF = PointPairFeatures()
    
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
            edge_dim=4,
            smoothing=None,
            name='net_conv_edge_pool',
            lin=True,
            crf=False
        ):
        super(NetConvEdgePool, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.use_skips = use_skips
        self.sum_skips = sum_skips
        self.num_top_convs = num_top_convs
        self.num_bottom_convs = num_bottom_convs*(depth > 0)
        self.edge_dim = edge_dim
        self.smoothing = smoothing
        self.name = name
        self.lin = lin
        self.crf = crf
        if(act == 'relu'):
            self.act = F.relu
        elif(act == 'elu'):
            self.act = F.elu
        elif(act == 'selu'):
            self.act = F.selu
        
        # decide what operations we need to do for each pool
        self.conv_type = conv_args["name"]
        if(depth > 0):
            self.pool_type = pool_args["name"]
        if(self.conv_type == "FeaSt"):
            self.update_mesh = False
            self.edge_features = False
            self.scale_edge_features = False
        elif(self.conv_type == "Spline"):
            self.update_mesh = True
            self.edge_features = True
            self.scale_edge_features = "clip"
        elif(self.conv_type == "GMM"):
            self.update_mesh = True
            self.edge_features = True
            self.scale_edge_features = "norm"
        elif(self.conv_type == "NN"):
            self.update_mesh = True
            self.edge_features = True
            self.scale_edge_features = "norm"
        elif(self.conv_type == "PPF"):
            self.update_mesh = True
            self.edge_features = False
            self.scale_edge_features = False
        elif(self.conv_type == "CG"):
            self.update_mesh = True
            self.edge_features = True
            self.scale_edge_features = "norm"
        
        # build transforms
        if(self.update_mesh):
            self.gmn = GenerateMeshNormals()
        if(self.edge_features):
            self.ppf = PointPairFeatures()
        
        # containers to hold the layers
        self.top_convs = torch.nn.ModuleList()
        self.down_convs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.bottom_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        
        # set size of every layer
        num_lin = 4 # 1 + 3
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
        
        if crf:
            self.crf1 = ContinuousCRF(**crf_args)
        
        if lin:
            # up FC layers
            self.lin2 = nn.Linear(nhidden[-3], nhidden[-2])
            self.lin3 = nn.Linear(nhidden[-2], nhidden[-1])
            self.lin4 = nn.Linear(nhidden[-1], nOut)
    
    def makePool(self, nin, name=None, **kwargs):
        # Edge Pool
        if(name == "EdgePool"):
            return EdgePooling(nin, **kwargs)
        # Edge Pool with edge features
        if(name == "EdgePool_EF"):
            return EdgePooling_EF(nin, self.edge_dim, **kwargs)
    
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
        if(self.pool_type == 'EdgePool'):
            return (x, data.edge_index, data.batch)
        elif(self.pool_type == 'EdgePool_EF'):
            return (x, data.edge_index, data.edge_attr, data.batch)
    
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
    
    def getpooledMesh(self, data, unpool_info, new_x, new_edge_index, new_batch):
        if(self.update_mesh):
            # update the node positions and faces
            ind1 = torch.empty_like(new_batch, device=torch.device('cpu'))
            ind2 = torch.empty_like(new_batch, device=torch.device('cpu'))
            
            N = unpool_info.cluster.size(0)
            #for i in range(N):
                #ind1a[unpool_info.cluster[i]] = i
                #ind2a[unpool_info.cluster[N-1-i]] = N-1-i
            ind1[unpool_info.cluster] = torch.arange(N)
            ind2[unpool_info.cluster.flip((0))] = torch.arange(N-1, -1, -1)
            new_pos = (data.pos[ind1] + data.pos[ind2])/2
            
            new_face = torch.empty_like(data.face)
            new_face[0,:] = unpool_info.cluster[data.face[0,:]]
            new_face[1,:] = unpool_info.cluster[data.face[1,:]]
            new_face[2,:] = unpool_info.cluster[data.face[2,:]]
            fi = (new_face[0,:] == new_face[1,:]) + (new_face[0,:] == new_face[2,:]) + (new_face[1,:] == new_face[2,:])
            new_face = new_face[:,~fi]
            
            new_data = Data(x=new_x, edge_index=new_edge_index, pos=new_pos, face=new_face, batch=new_batch)
            new_data.unpool_info = unpool_info
            
            if self.edge_features:
                if new_data.norm is None:
                    self.GMN(new_data)
                    self.PPF(new_data)
                
                # scale edge features to lie within [0,1]
                if self.scale_edge_features == "clip":
                    e_mean = data.edge_attr.mean(axis=0)
                    e_std = data.edge_attr.std(axis=0)
                    e_min = e_mean - 2*e_std
                    e_max = e_mean + 2*e_std
                    data.edge_attr = torch.clamp((data.edge_attr - e_min)/(e_max - e_min), min=0.0, max=1.0)
                elif self.scale_edge_features == "norm":
                    e_mean = data.edge_attr.mean(axis=0)
                    e_std = data.edge_attr.std(axis=0)
                    data.edge_attr = (data.edge_attr - e_mean)/e_std
            else:
                if new_data.norm is None:
                    self.PPF(new_data)
        else:
            new_data = Data(x=new_x, edge_index=new_edge_index, batch=new_batch)
            new_data.unpool_info = unpool_info
        
        return new_data
    
    def forward(self, data, debug=False):
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
            if(self.use_skips):
                skip_connections.append(x)
            
            # pooling 
            args = self.getPoolArgs(x, graph_activations[i])
            x, edge_index, batch, unpool = self.down_pools[i](*args)
            data_pooled = self.getpooledMesh(graph_activations[i], unpool, x, edge_index, batch)
            graph_activations.append(data_pooled)
        
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
        
        #del skip_connections
        #del graph_activations
        
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
