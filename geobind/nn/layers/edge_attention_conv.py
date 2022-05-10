from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax, geodesic_distance

def get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    return torch.atan2(
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))

def edge_pair_features(
        edge_index: Tensor,
        pos: Tensor,
        normal: Tensor,
        faces: Optional[Tensor] = None,
        normalize_gd: bool = True
    ) -> Tensor:
    
    pseudo = pos[edge_index[0]] - pos[edge_index[1]]
    
    features = [
        pseudo.norm(p=2, dim=1),
        get_angle(normal[edge_index[1]], pseudo),
        get_angle(normal[edge_index[0]], pseudo),
        get_angle(normal[edge_index[1]], normal[edge_index[0]])
    ]
    if faces is not None:
        g = geodesic_distance(pos, faces, src=edge_index[0], dest=edge_index[1], norm=False, num_workers=0)
        if normalize_gd:
            g = (g + 1e-8)/(features[0] + 1e-8)
        features.append(g)
    
    return torch.stack(features, dim=1)

class EdgeAttentionConv(MessagePassing):
    r"""An edge-attention convolution based on the The PPFNet operator from the `"PPFNet: Global Context Aware Local
    Features for Robust 3D Point Matching" <https://arxiv.org/abs/1802.02669>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j, \|
        \mathbf{d_{j,i}} \|, \angle(\mathbf{n}_i, \mathbf{d_{j,i}}),
        \angle(\mathbf{n}_j, \mathbf{d_{j,i}}), \angle(\mathbf{n}_i,
        \mathbf{n}_j) \right)

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote neural networks, *.i.e.* MLPs, which takes in node features and
    :class:`torch_geometric.transforms.PointPairFeatures`.

    Args:
        local_nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` and
            relative spatial coordinates :obj:`pos_j - pos_i` of shape
            :obj:`[-1, in_channels + num_dimensions]` to shape
            :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): A neural network
            :math:`\gamma_{\mathbf{\Theta}}` that maps aggregated node features
            of shape :obj:`[-1, out_channels]` to shape :obj:`[-1,
            final_out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          positions :math:`(|\mathcal{V}|, 3)` or
          :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
          point normals :math:`(|\mathcal{V}, 3)` or
          :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

    """
    def __init__(self, 
            gate_nn: Callable, 
            local_nn: Optional[Callable] = None,
            global_nn: Optional[Callable] = None,
            edge_bn: Optional[Callable] = None,
            add_self_loops: bool = False, 
            attention_type: str = "local",
            combine_vertex_features: str = "cat",
            weighted_average: bool = False,
            use_geodesic_distance: bool = False,
            normalize_gd: bool = True,
            use_edge_features=True,
            **kwargs
        ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.gate_nn = gate_nn
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.edge_bn = edge_bn
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.weighted_average = weighted_average
        self.use_geodesic_distance = use_geodesic_distance
        self.combine_vertex_features = combine_vertex_features
        self.normalize_gd = normalize_gd
        self.use_edge_features = use_edge_features
        if weighted_average:
            assert attention_type == "local", "weighted average can only be used with local attention!"
        
        # how to construct edge message
        if self.combine_vertex_features == "cat":
            self.msg_fn = lambda xi, xj, eij: torch.cat([xi, xj], dim=1) if eij is None else torch.cat([xi, xj, eij], dim=1)
        elif self.combine_vertex_features == "diff":
            self.msg_fn = lambda xi, xj, eij: xj - xi if eij is None else torch.cat([xj - xi, eij], dim=1)
        elif self.combine_vertex_features == "abs_diff":
            self.msg_fn = lambda xi, xj, eij: torch.abs(xj - xi) if eij is None else torch.cat([torch.abs(xj - xi), eij], dim=1)
        elif self.combine_vertex_features == "mean":
            self.msg_fn = lambda xi, xj, eij: (xi + xj)/2 if eij is None else torch.cat([(xi + xj)/2, eij], dim=1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)
        reset(self.gate_nn)
    
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        normal: Union[Tensor, PairTensor],
        edge_index: Adj, 
        faces: Optional[Tensor] = None,
        size: Optional[int] = None
    ) -> Tensor:
        """"""
        # propagate_type: (x: PairTensor, edge_attr: Tensor)  # noqa
        if not isinstance(x, tuple):
            x: PairTensor = (x, x)
        
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        
        if self.use_edge_features:
            # get edge features
            if self.use_geodesic_distance:
                edge_attr = edge_pair_features(edge_index, pos, normal, faces=faces, normalize_gd=self.normalize_gd)
            else:
                edge_attr = edge_pair_features(edge_index, pos, normal)
            if self.edge_bn is not None:
                edge_attr = self.edge_bn(edge_attr)
        else:
            edge_attr = None
        
        # do message-passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        if self.weighted_average:
            out = out[:,:-1] / (out[:,-1].view(-1, 1) + 1e-8)
        
        if self.global_nn is not None:
            #out = self.global_nn(torch.cat([x[0], out], dim=1))
            out = self.global_nn(out)
        
        return out

    def message(self, 
            x_i: Tensor,
            x_j: Tensor,
            index: Tensor,
            edge_attr: Optional[Tensor] = None
        ) -> Tensor:
        
        # construct edge message
        msg = self.msg_fn(x_i, x_j, edge_attr)
        gate = self.gate_nn(msg)
        if gate.dim() == 1:
            gate = gate.view(-1,1)
        assert gate.dim() == msg.dim() and gate.size(0) == msg.size(0)
        
        # apply activation
        if self.attention_type == "global":
            gate = softmax(gate, index)
        elif self.attention_type == "local":
            gate = F.hardsigmoid(gate)
        
        # update message
        if self.local_nn is not None:
            msg = self.local_nn(x_j)
        else:
            msg = x_j
        msg = gate * msg
        
        if self.weighted_average:
            return torch.cat([msg, gate], dim=1)
        else:
            return msg
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')
