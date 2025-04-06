#!/usr/bin/env python
# third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.aggr import MultiAggregation

# pnabind modules
from pnabind.nn.layers import SAModule, FPModule, ContinuousCRF
from pnabind.nn.utils import MLP

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

def makePPFConv(dim_in, dim_out, conv_args, act="relu"):
    from torch_geometric.nn import PPFConv
    
    conv_args.setdefault("nhidden", 16)
    conv_args.setdefault("local_depth", 3)
    conv_args.setdefault('aggr', 'max')
    conv_args.setdefault('batch_norm', False)
    
    dims = [dim_in + 4] + [conv_args['nhidden']]*conv_args['local_depth']
    nn_local = MLP(dims,
        batch_norm=conv_args['batch_norm'],
        bn_kwargs=bn_kwargs,
        act=act,
        batchnorm_position="left"
    )
    
    dims = [dims[-1], dim_out]
    nn_global = MLP(dims, 
        batch_norm=False,
        act=act
    )
    conv = PPFConv(nn_local, nn_global, add_self_loops=False)
    conv.aggr = conv_args['aggr']
    
    forward_args = lambda x, pos, norm, edge_index, edge_attr: (x, pos, norm, edge_index)
    
    return conv, forward_args

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

def makeTransformerConv(dim_in, dim_out, edge_dim, conv_args):
    from torch_geometric.nn import TransformerConv
    conv_args.setdefault("kwargs", {})
    conv_args["kwargs"].setdefault("heads", 1)
    conv_args["kwargs"].setdefault("concat", False)
    conv_args["kwargs"]["edge_dim"] = edge_dim
    
    conv = TransformerConv(dim_in, dim_out, **conv_args["kwargs"])
    forward_args = lambda x, pos, norm, edge_index, edge_attr: (x, edge_index, edge_attr)
    
    return conv, forward_args

def makeGMMConv(dim_in, dim_out, edge_dim, conv_args):
    from torch_geometric.nn import GMMConv
    conv_args.setdefault("kernel_size", 6)
    conv_args.setdefault("kwargs", {})
    conv_args["kwargs"].setdefault("aggr", "mean")
    conv_args["kwargs"].setdefault("separate_gaussians", False)
    conv_args["kwargs"].setdefault("root_weight", True)
    
    if conv_args["kwargs"]["aggr"] == "multi":
        aggr = MultiAggregation(['mean', 'std', 'min', 'max'],
            mode="proj",
            mode_kwargs={
                "in_channels": dim_out,
                "out_channels": dim_out
            }
        )
        conv_args["kwargs"]["aggr"] = aggr
    
    conv = GMMConv(dim_in, dim_out, edge_dim, conv_args["kernel_size"], **conv_args["kwargs"])
    forward_args = lambda x, pos, norm, edge_index, edge_attr: (x, edge_index, edge_attr)
    
    return conv, forward_args

def makeEdgeAttnConv(dim_in, dim_out, edge_dim, conv_args,
        act="relu"
    ):
    from pnabind.nn.layers import EdgeAttentionConv
    # Set defaults
    conv_args.setdefault("gate_depth", 2)
    conv_args.setdefault("local_depth", 2)
    conv_args.setdefault("global_depth", 2)
    conv_args.setdefault("num_heads", 1)
    conv_args.setdefault("use_global", True)
    conv_args.setdefault("use_local", False)
    conv_args.setdefault("batch_norm", True)
    
    conv_args["kwargs"].setdefault("attention_type", "softmax")
    conv_args["kwargs"].setdefault("average_heads", False)
    conv_args["kwargs"].setdefault("aggr", "sum")
    conv_args["kwargs"]["add_self_loops"] = False
    
    
    # Gate NN
    dims = [edge_dim]*conv_args['gate_depth'] + [conv_args['num_heads']]
    acts = [act]*(conv_args['gate_depth'] - 1) + [None]
    gate_nn = MLP(dims, 
        act=acts,
        batch_norm=conv_args['batch_norm']
    )
    
    if conv_args["use_local"]:
        # local NN
        dims = [dim_in]*(conv_args['local_depth'] + 1)
        local_nn = MLP(dims,
            batch_norm=conv_args['batch_norm'],
            act=act
        )
    else:
        local_nn = None
    
    if conv_args["use_global"]:
        # global NN
        if conv_args["kwargs"]["average_heads"]:
            dims = [dim_in] + [dim_out]*conv_args['global_depth']
        else:
            dims = [dim_in*conv_args["num_heads"]] + [dim_out]*conv_args['global_depth']
        global_nn = MLP(dims,
            batch_norm=conv_args['batch_norm'],
            act=act
        )
    else:
        global_nn = None
    
    conv = EdgeAttentionConv(gate_nn, local_nn=local_nn, global_nn=global_nn, **conv_args["kwargs"])
    forward_args = lambda x, pos, norm, edge_index, edge_attr: (x, edge_index, edge_attr)
    
    return conv, forward_args

def makeXConv(dim_in, dim_out, conv_args):
    from torch_geometric.nn import XConv

    conv_args.setdefault("kwargs", {})
    conv_args.setdefault("kernel_size", 16)
    conv_args["kwargs"].setdefault("dilation", 2)
    conv_args["kwargs"].setdefault("num_workers", 8)
    
    conv = XConv(dim_in, dim_out, 3, conv_args["kernel_size"], **conv_args["kwargs"])
    forward_args = lambda x, pos, norm, edge_index, edge_attr: (x, pos)
    
    return conv, forward_args

def makeFPMLP(dim_in, dim_out, nSkip, batch_norm=False, bn_kwargs={}, act='relu'):
    dim = dim_in + nSkip
    
    return MLP([dim, dim, dim_out], batch_norm=batch_norm, bn_kwargs=bn_kwargs, act=act)

class AuxModel(torch.nn.Module):
    def __init__(self, dim_in,
        nhidden=16,
        depth=4,
        act='silu'
    ):
        super(AuxModel, self).__init__()
        from torch_geometric.nn import SAGEConv
        self.depth = depth
        self.act = F.silu
        
        dim = dim_in
        self.convs = torch.nn.ModuleList()
        for i in range(depth):
            conv = SAGEConv(dim, nhidden, aggr="sum")
            self.convs.append(conv)
            dim = nhidden
        
        self.lin_out = MLP(
            [dim, 2*dim, 2*dim, 1],
            batch_norm=False,
            act=[act, act, None]
        )
        
    def forward(self, x, edge_index):
        for i in range(self.depth):
            x = self.convs[i](x, edge_index)
            x = self.act(x)
        x = self.lin_out(x)
        
        return x

class Model(torch.nn.Module):
    def __init__(self, dim_in, dim_out,
            conv_args=None,
            depth=3,
            lin_depth=2,
            lin_hidden=None,
            nhidden=16,
            ratios=None,
            radii=None,
            max_neighbors=64,
            knn_num=3,
            batch_norm=True,
            name="model",
            act="relu",
            use_lin=True,
            use_crf=False,
            use_sa_residuals=False,
            use_aux_gnn=False
        ):
        super(Model, self).__init__()
        
        ### Set up ###
        self.depth = depth
        self.name = name
        self.act = act
        self.use_lin = use_lin
        self.use_crf = use_crf
        self.use_sa_residuals = use_sa_residuals
        self.use_aux_gnn = use_aux_gnn
        
        if ratios is None:
            ratios = [0.5]*depth
        assert len(ratios) == depth
        
        if radii is None:
            radii = [None]*depth
        assert len(radii) == depth
        
        ### Model Layers ###
        # linear input
        if use_lin:
            if lin_hidden:
                dims = [dim_in] + [lin_hidden]*(lin_depth-1) + [nhidden]
            else:
                dims = [dim_in]*lin_depth + [nhidden]
            self.lin_in = MLP(dims, batch_norm=batch_norm, act=act)
        else:
            nhidden = dim_in
        
        # pooling/unpooling layers
        if use_sa_residuals:
            self.residual_mlps = torch.nn.ModuleList()
        self.SA_modules = torch.nn.ModuleList()
        self.FP_modules = torch.nn.ModuleList()
        for i in range(depth):
            # conv/pool
            if conv_args["type"] == "TransformerConv":
                edge_features = conv_args.get("edge_features", None)
                dim_edge = EDGE_DIM[edge_features](nhidden)
                
                sa_kwargs = dict(
                    edge_features=edge_features,
                    dim_edge=dim_edge
                )
                a = (nhidden, nhidden, dim_edge, conv_args)
                k = {}
                f = makeTransformerConv
            elif conv_args["type"] == "GMMConv":
                edge_features = conv_args.get("edge_features", "ppf")
                dim_edge = EDGE_DIM[edge_features](nhidden)
                
                dims = [dim_edge, conv_args.get("edge_proj_hidden", 64), 4]
                sa_kwargs = dict(
                    edge_features=edge_features,
                    dim_edge=dim_edge,
                    edge_projection=True,
                    edge_mlp_dims=dims,
                    act=act,
                    apply_act=conv_args.get("apply_activation", True)
                )
                a = (nhidden, nhidden, 4, conv_args)
                k = {}
                f = makeGMMConv
            elif conv_args["type"] == "PPFConv":
                edge_features = None
                dim_edge = EDGE_DIM[edge_features](nhidden)
                
                sa_kwargs = dict(
                    edge_features=edge_features,
                    dim_edge=dim_edge
                )
                a = (nhidden, nhidden, conv_args)
                k = {"act": act}
                f = makePPFConv
            elif conv_args["type"] == "CGConv":
                edge_features = conv_args.get("edge_features", "ppf")
                dim_edge = EDGE_DIM[edge_features](nhidden)
                
                sa_kwargs = dict(
                    edge_features=edge_features,
                    dim_edge=dim_edge,
                    apply_act=False
                )
                a = (nhidden, dim_edge, conv_args)
                k = {}
                f = makeCGConv
            elif conv_args["type"] == "EdgeAttentionConv":
                edge_features = conv_args.get("edge_features", "ppf")
                dim_edge = EDGE_DIM[edge_features](nhidden)
                
                sa_kwargs = dict(
                    edge_features=edge_features,
                    dim_edge=dim_edge
                )
                a = (nhidden, nhidden, dim_edge, conv_args)
                k = {"act": act}
                f = makeEdgeAttnConv
            elif conv_args["type"] == "XConv":
                sa_kwargs = dict(radial_graph=False)
                a = (nhidden, nhidden, conv_args)
                k = {}
                f = makeXConv
            
            # add convolutions/SA module
            convs = []
            for j in range(conv_args.get("num", 1)):
                conv, args_fn = f(*a, **k)
                convs.append(conv)
            
            self.SA_modules.append(SAModule(convs, args_fn, ratios[i], 
                radius=radii[i],
                max_neighbors=max_neighbors,
                use_edge_bn=batch_norm,
                **sa_kwargs
            ))
            
            # unpool
            mlp = makeFPMLP(nhidden, nhidden, nhidden, batch_norm=batch_norm, act=act)
            self.FP_modules.append(
                FPModule(mlp, k=knn_num)
            )
            
            if use_sa_residuals:
                 self.residual_mlps.append(
                    MLP([nhidden, 2*nhidden, dim_out], batch_norm=[batch_norm, False], act=[act, None])
                 )
        self.nout = nhidden
        
        # linear layers
        self.lin_out = MLP(
            [nhidden, nhidden, dim_out],
            batch_norm=[batch_norm, False],
            act=[act, None]
        )
        self.nout = dim_out
        if use_aux_gnn:
            self.aux_gnn = AuxModel(nhidden, nhidden=nhidden)
        
        # CRF layers
        if use_crf:
            self.crf1 = ContinuousCRF()
            self.crf2 = ContinuousCRF()
    
    def forward(self, data, return_activations=False):
        if return_activations:
            idxs = []
        
        if self.use_sa_residuals and self.training:
            residuals = []
        
        # lin in
        if self.use_lin:
            x = self.lin_in(data.x)
        else:
            x = data.x
        
        if self.use_crf:
            x = self.crf1(x, data.edge_index)
        
        # conv/pooling
        norm = data.norm
        sa_outs = [(x, data.pos, data.batch)]
        for i in range(self.depth):
            args = (*sa_outs[i], norm)
            x, pos, batch, idx = self.SA_modules[i].forward(*args)
            sa_outs.append((x, pos, batch))
            norm = norm[idx]
            if return_activations:
                idxs.append(idx.cpu().numpy())
            
            if self.use_sa_residuals and self.training:
                xr = self.residual_mlps[i](x)
                residuals.append( (xr, idx) )
        
        # unpooling
        fp_out = self.FP_modules[-1].forward(*sa_outs[-1], *sa_outs[-2])
        if return_activations:
            fp_outs = [fp_out]
        for i in range(1, self.depth):
            j = - i
            fp_out = self.FP_modules[j-1].forward(*fp_out, *sa_outs[j-2])
            if return_activations:
                fp_outs.append(fp_out)
        x = fp_out[0]
        
        if self.use_aux_gnn and self.training:
            xr = self.aux_gnn(x, data.edge_index)
            residuals.append( (xr, None) )
        
        if self.use_crf:
            x = self.crf2(x, data.edge_index)
        
        # lin out
        x = self.lin_out(x)
        
        if self.training:
            if self.use_sa_residuals:
                return x, residuals
            else:
                return  x
        else:
            if return_activations:
                return x, fp_outs, sa_outs, idxs
            else:
                return  x
    
