from typing import Optional, Callable, Union
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor, Adj

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.conv import PPFConv

def get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    return torch.atan2(
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))


def point_pair_features(pos_i: Tensor, pos_j: Tensor, normal_i: Tensor,
                        normal_j: Tensor) -> Tensor:
    pseudo = pos_j - pos_i
    return torch.stack([
        pseudo.norm(p=2, dim=1),
        get_angle(normal_i, pseudo),
        get_angle(normal_j, pseudo),
        get_angle(normal_i, normal_j)
    ], dim=1)


class PPHConv(PPFConv):
    r"""The PPFNet operator from the `"PPFNet: Global Context Aware Local
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
    """
    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, 
                 kernel_type: str = 'rbf', 
                 num_bins: int = 8,
                 normalize: bool = False,
                 eps: float = 1e-5,
                 aggr: str = 'add', **kwargs):
        kwargs.setdefault('node_dim', 0)
        super(PPHConv, self).__init__(local_nn, global_nn, add_self_loops=add_self_loops, **kwargs)
        
        self.aggr = aggr
        self.kernel_type = kernel_type
        self.normalize = normalize
        self.eps = eps
        
        # set bin centers
        self.num_bins = num_bins
        w = 2/num_bins
        self.bin_c = Parameter( torch.linspace(-2, 2, num_bins, dtype=torch.float32) )
        self.bin_w = Parameter( torch.full(self.bin_c.size(), 1/w ,dtype=torch.float32) )

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor],
                normal: Union[Tensor, PairTensor],
                edge_index: Adj) -> Tensor:  # yapf: disable
        """"""
        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if isinstance(normal, Tensor):
            normal: PairTensor = (normal, normal)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor, normal: PairTensor)  # noqa
        out = self.propagate(edge_index, x=x, pos=pos, normal=normal,
                             size=None)
        s = out.size()
        if self.normalize:
            out = out / (out.sum(dim=-1).view(-1, s[1], 1) + self.eps)
        
        if self.global_nn is not None:
            # flatten histogram and perform transform
            out = self.global_nn(out.view(-1, s[1]*s[2]))
        
        return out

    def message(self, x_j: OptTensor, pos_i: Tensor, pos_j: Tensor,
                normal_i: Tensor, normal_j: Tensor) -> Tensor:
        msg = point_pair_features(pos_i, pos_j, normal_i, normal_j)
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        
        # compute histogram
        s = msg.size()
        msg = torch.exp(-((msg.view(s[0], s[1], 1) - self.bin_c)*self.bin_w)**2)
        
        return msg

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
                                                      self.local_nn,
                                                      self.global_nn)
