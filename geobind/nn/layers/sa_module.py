# third party modules
import torch
from torch import Tensor
from torch.nn import ReLU, ELU, Identity, Tanh, Dropout, PReLU, GELU, SiLU

from torch_geometric.nn import radius, fps
from geobind.nn.utils import MLP

ACTIVATION = {
    'relu': ReLU,
    'elu': ELU,
    'tanh': Tanh,
    'prelu': PReLU,
    'gelu': GELU,
    'silu': SiLU
}

def get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    return torch.atan2(
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))

def point_pair_features(
        edge_index: Tensor,
        pos: Tensor,
        normal: Tensor
    ) -> Tensor:
    
    pseudo = pos[edge_index[0]] - pos[edge_index[1]]
    
    features = [
        pseudo.norm(p=2, dim=1),
        get_angle(normal[edge_index[1]], pseudo),
        get_angle(normal[edge_index[0]], pseudo),
        get_angle(normal[edge_index[1]], normal[edge_index[0]])
    ]
    
    return torch.stack(features, dim=1)

class SAModule(torch.nn.Module):
    def __init__(self, conv, conv_arg_fn, ratio,
            radius=None,
            max_neighbors=32,
            edge_features=None,
            use_edge_bn=False,
            dim_edge=None,
            radial_graph=True,
            edge_projection=None,
            edge_mlp_dims=None,
            act=None,
            apply_act=False
        ):
        super(SAModule, self).__init__()
        """This module acts as a pooling/conv layer. Taken from pytorch-geometric examples with modifications."""
        self.ratio = ratio
        self.r = radius
        self.K = max_neighbors
        self.get_conv_args = conv_arg_fn
        self.radial_graph = radial_graph
        self.edge_projection = edge_projection
        self.use_edge_bn = use_edge_bn
        self.apply_act = apply_act
        
        # list of convolution layers
        if not isinstance(conv, list):
            conv = [conv]
        self.conv = torch.nn.ModuleList(conv)
        
        # how to construct edge features
        if edge_features == "cat+ppf":
            self.edge_fn = lambda x, pos, normal, e: torch.cat([x[e[1]], x[e[0]], point_pair_features(e, pos, normal)], dim=1)
        elif edge_features == "diff+ppf":
            self.edge_fn = lambda x, pos, normal, e: torch.cat([x[e[0]] - x[e[1]], point_pair_features(e, pos, normal)], dim=1)
        elif edge_features == "abs_diff+ppf":
            self.edge_fn = lambda x, pos, normal, e: torch.cat([torch.abs(x[e[0]] - x[e[1]]), point_pair_features(e, pos, normal)], dim=1)
        elif edge_features == "mean+ppf":
            self.edge_fn = lambda x, pos, normal, e: torch.cat([(x[e[0]] + x[e[1]])/2, point_pair_features(e, pos, normal)], dim=1)
        elif edge_features == "cat":
            self.edge_fn = lambda x, pos, normal, e: torch.cat([x[e[1]], x[e[0]]], dim=1)
        elif edge_features == "diff":
            self.edge_fn = lambda x, pos, normal, e: x[e[0]] - x[e[1]]
        elif edge_features == "abs_diff":
            self.edge_fn = lambda x, pos, nomral, e: torch.abs(x[e[0]] - x[e[1]])
        elif edge_features == "mean":
            self.edge_fn = lambda x, pos, normal, e: (x[e[0]] + x[e[1]])/2
        elif edge_features == "ppf":
            self.edge_fn = lambda x, pos, normal, e: point_pair_features(e, pos, normal)
        else:
            self.edge_fn = lambda x, pos, normal, e: None
        
        if (edge_features is not None) and use_edge_bn:
            assert dim_edge is not None
            self.edge_bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim_edge) for _ in range(len(self.conv))])
        else:
            self.use_edge_bn = False
        
        if self.apply_act:
            self.acts = torch.nn.ModuleList([ACTIVATION[act]() for _ in range(len(conv))])
        
        if self.edge_projection:
            assert edge_mlp_dims is not None
            self.edge_mlps = torch.nn.ModuleList([MLP(edge_mlp_dims, batch_norm=False, act=act) for _ in range(len(conv))])
    
    def forward(self, x, pos, batch, norm, permute_edge_features=False):
        if self.radial_graph:
            # pool points based on FPS algorithm, returning num_points*ratio centroids
            idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
            
            # finds points within radius `self.r` of the centroids, up to `self.K` pts per centroid
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.K)
            
            # perform convolution over edges joining centroids to their neighbors within ball of radius `self.r`
            row = idx[row] # re-index to node numbers in full graph
            edge_index = torch.stack([col, row], dim=0)
            
            for i in range(len(self.conv)):
                # get edge features if requested
                edge_attr = self.edge_fn(x, pos, norm, edge_index)
                
                if self.use_edge_bn:
                    edge_attr = self.edge_bn[i](edge_attr)
                
                if permute_edge_features:
                    perm = torch.randperm(edge_attr.size(0))
                    edge_attr = edge_attr[perm]
                    
                if self.edge_projection:
                    edge_attr = self.edge_mlps[i](edge_attr)
                
                # perform convolution
                args = self.get_conv_args(x, pos, norm, edge_index, edge_attr)
                x = self.conv[i](*args)
                
                if self.apply_act:
                    x = self.acts[i](x)
            
            # slice tensors with centroid indices
            x, pos, batch = x[idx], pos[idx], batch[idx]
            
            return x, pos, batch, idx
        else:
            for i in range(len(self.conv)):
                args = self.get_conv_args(x, pos, norm, None, None)
                x = self.conv[i](*args, batch=batch)
            
            idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
            
            return x[idx], pos[idx], batch[idx], idx
