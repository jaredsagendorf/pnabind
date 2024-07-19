import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Dropout

from pnabind.nn.utils import MLP
from pnabind.nn.layers import LocalAttentionPooling, SAModule
from torch.nn import Sequential as Seq
from torch_geometric.nn.aggr import MultiAggregation

# Edge features dimensionality
EDGE_DIM = {
    "cat+ppf": lambda dim_in: 2*dim_in + 4,
    "diff+ppf": lambda dim_in: dim_in + 4,
    "abs_diff+ppf":  lambda dim_in: dim_in + 4,
    "mean+ppf":  lambda dim_in: dim_in + 4,
    "cat": lambda dim_in: 2*dim_in,
    "diff": lambda dim_in: dim_in,
    "abs_diff": lambda dim_in: dim_in,
    "mean": lambda dim_in: dim_in,
    "ppf": lambda dim_in: 4,
    None: lambda dim_in: 0
}

def makeCGConv(dim_in, edge_dim, conv_args):
    from torch_geometric.nn import CGConv
    conv_args.setdefault("kwargs", {})
    conv_args["kwargs"].setdefault("aggr", "mean")
    conv_args["kwargs"].setdefault("batch_norm", False)
    
    if conv_args["kwargs"]["aggr"] == "multi":
        aggr = MultiAggregation(['mean', 'std', 'min', 'max'],
            mode="proj",
            mode_kwargs={
                "in_channels": dim_in,
                "out_channels": dim_in
            }
        )
        conv_args["kwargs"]["aggr"] = aggr
        
    conv = CGConv(dim_in, dim=edge_dim, **conv_args["kwargs"])
    forward_args = lambda x, pos, norm, edge_index, edge_attr: (x, edge_index, edge_attr)
    
    return conv, forward_args

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nIn, nOut, pool_args, bias=True):
        super(GlobalSAModule, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.pool_type = pool_args["name"]
        pool_args.setdefault("kwargs", {})
        
        if pool_args["name"] == "global_attention_pool":
            from torch_geometric.nn import GlobalAttention
            self.gate_nn = MLP([nIn, nIn, 1], batch_norm=False, dropout=pool_args.get("dropout", 0.0), dropout_position="left", bias=bias, act=['relu', None])
            self.nn = nn.Identity()
            self.pool = GlobalAttention(self.gate_nn, self.nn, **pool_args["kwargs"])
        elif pool_args["name"] == "local_attention_pool":
            self.gate_nn = MLP([nIn, nIn, 1], batch_norm=False, dropout=pool_args.get("dropout", 0.0), dropout_position="left", bias=bias, act=['relu', None])
            self.nn = nn.Identity()
            self.pool = LocalAttentionPooling(self.gate_nn, self.nn, **pool_args["kwargs"])
        
    def forward(self, x, pos, batch, edge_index=None, return_attn=False):
        x = self.pool(x, batch, return_attn=return_attn)
        
        return x

class Model(torch.nn.Module):
    def __init__(self, nIn, nout=None,
            pool_args=None,
            conv_args=None,
            nhidden=16,
            depth=3,
            ratios=None,
            radii=None,
            max_neighbors=64,
            name='classifier',
            dropout=0.0,
            batch_norm=True,
            bias=True
        ):
        super(Model, self).__init__()
        
        ### Set up ###
        self.depth = depth
        self.name = name
        self.dropout = dropout
        
        ### Model Layers ###
        # linear input
        if batch_norm:
            self.lin_in = MLP([nIn, 64, nhidden], batch_norm=True, bias=bias)
        else:
            self.lin_in = MLP([nIn, 64, nhidden], batch_norm=False, dropout=dropout, bias=bias)
        
        # pooling layers
        self.GP_module = GlobalSAModule(nhidden, nhidden, pool_args, bias=bias)
        self.SA_modules = torch.nn.ModuleList()
        
        edge_features = conv_args.get("edge_features", "ppf")
        dim_edge = EDGE_DIM[edge_features](nhidden)
        sa_kwargs = dict(
            edge_features=edge_features,
            dim_edge=dim_edge
        )
        
        for i in range(depth):
            # add convolutions/SA module
            convs = []
            for j in range(conv_args.get("num", 1)):
                conv, args_fn = makeCGConv(nhidden, dim_edge, conv_args)
                convs.append(conv)
            
            self.SA_modules.append(SAModule(convs, args_fn, ratios[i], 
                radius=radii[i],
                max_neighbors=max_neighbors,
                use_edge_bn=batch_norm,
                **sa_kwargs
            ))
        
        # linear layers
        if batch_norm:
            self.lin_out = MLP(
                [nhidden, nhidden, nout],
                batch_norm=[True, False],
                act=['relu', None],
                bias=bias
            )
        else:
            self.lin_out = MLP(
                [nhidden, nhidden, nout],
                dropout=[dropout, 0.0],
                act=['relu', None],
                bias=bias
            )
        self.nout = nout
    
    def forward(self, x, pos, norm, batch, y=None, return_attn=False, return_y=False, return_sa=False, zero_pooling=False, permute_edge_features=False):
        # lin1
        x = self.lin_in(x)
        
        # conv/pooling
        sa_outs = [(x, pos, batch)]
        for i in range(self.depth):
            args = (*sa_outs[i], norm)
            x, pos, batch, idx = self.SA_modules[i].forward(*args, permute_edge_features=permute_edge_features)
            sa_outs.append((x, pos, batch))
            norm = norm[idx]
            if return_y:
                y = y[idx]
        
        # global pooling
        if return_attn:
            x, attn = self.GP_module(*sa_outs[-1], return_attn=return_attn)
        else:
            x = self.GP_module(*sa_outs[-1])
        if zero_pooling:
            x = 0*x
        
        # lin2
        x = F.dropout(x, self.dropout, self.training)
        x = self.lin_out(x)
        
        args = [x]
        if return_attn:
            args.append(attn)
        if return_y:
            args.append(y)
        if return_sa:
            args.append(sa_outs[-1])
        
        return args[0] if (len(args) == 1) else tuple(args)
