from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.inits import zeros, glorot
from .get_edge_features import edge_pair_features

class MMConv(MessagePassing):
    r"""The gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \frac{1}{K} \sum_{k=1}^K
        \mathbf{w}_k(\mathbf{e}_{i,j}) \odot \mathbf{\Theta}_k \mathbf{x}_j,

    where

    .. math::
        \mathbf{w}_k(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left(
        \mathbf{e} - \mathbf{\mu}_k \right)}^{\top} \Sigma_k^{-1}
        \left( \mathbf{e} - \mathbf{\mu}_k \right) \right)

    denotes a weighting function based on trainable mean vector
    :math:`\mathbf{\mu}_k` and diagonal covariance matrix
    :math:`\mathbf{\Sigma}_k`.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int): Number of kernels :math:`K`.
        separate_gaussians (bool, optional): If set to :obj:`True`, will
            learn separate GMMs for every pair of input and output channel,
            inspired by traditional CNNs. (default: :obj:`False`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self,
            edge_nn: Callable,
            global_nn: Callable,
            out_channels: int,
            dim: int,
            kernel_size: int,
            edge_bn: Optional[Callable] = None,
            aggr: str = 'mean',
            diag_cov: bool = True,
            combine_vertex_features: str = 'cat',
            **kwargs
        ):
        super(GMMConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels # number of separate mixture models
        self.dim = dim # size of feature space
        self.kernel_size = kernel_size # number of mixture components
        self.diag_cov = diag_cov
        
        self.edge_nn = edge_nn
        self.edge_bn = edge_bn
        self.w = Parameter(
            torch.Tensor(out_channels, kernel_size)) # [M, K]
        self.mu = Parameter(
            torch.Tensor(out_channels, kernel_size, dim)) # [M, K, D]
        if diag_cov:
            self.sigma = Parameter(
                torch.Tensor(out_channels, kernel_size, dim)) # [M, K, D]
        else:
            self.Q = Parameter(
                torch.Tensor(out_channels, kernel_size, dim, dim)) # [M, K, D, D]
        
        # how to construct edge message
        if combine_vertex_features == "cat":
            self.msg_fn = lambda xi, xj, eij: torch.cat([xi, xj], dim=1) if eij is None else torch.cat([xi, xj, eij], dim=1)
        elif combine_vertex_features == "diff":
            self.msg_fn = lambda xi, xj, eij: xj - xi if eij is None else torch.cat([xj - xi, eij], dim=1)
        elif combine_vertex_features == "abs_diff":
            self.msg_fn = lambda xi, xj, eij: torch.abs(xj - xi) if eij is None else torch.cat([torch.abs(xj - xi), eij], dim=1)
        elif combine_vertex_features == "mean":
            self.msg_fn = lambda xi, xj, eij: (xi + xj)/2 if eij is None else torch.cat([(xi + xj)/2, eij], dim=1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.w)
        glorot(self.mu)
        glorot(self.sigma)
    
    def forward(self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        normal: Union[Tensor, PairTensor],
        edge_index: Adj,
        size: Optional[int] = None
    )-> Tensor:
        
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.use_edge_features:
            # get edge features
            edge_attr = edge_pair_features(edge_index, pos, normal)
            if self.edge_bn is not None:
                edge_attr = self.edge_bn(edge_attr)
        else:
            edge_attr = None
           
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                 size=size)
        out = self.global_nn(torch.cat([x[0], out], dim=1))
        
        return out
    
    def message(self, 
            x_i: Tensor,
            x_j: Tensor,
            index: Tensor,
            edge_attr: Optional[Tensor] = None
        ) -> Tensor:
        # construct edge message
        msg = self.msg_fn(x_i, x_j, edge_attr)
        msg = self.edge_nn(msg)
        
        # compute GMM sum
        EPS = 1e-15
        M = self.out_channels
        (E, D), K = msg.size(), self.kernel_size
        
        gaussian = -0.5 * (msg.view(E, 1, 1, D) - self.mu.view(1, M, K, D)).pow(2) # [E, M, K, D]
        gaussian = gaussian / ( EPS + self.sigma.view(1, M, K, D).pow(2) )
        gaussian = gaussian.sum(dim=-1) # sum over feature dimension -> [E, M, K]
        
        gaussian = self.w.view(1, M, K) * torch.exp(gaussian) # [E, M, K] 
        gaussian = gaussian.sum(dim=-1)  # [E, M]
        
        return gaussian
    
    def __repr__(self):
        return '{}({}, {}, dim={})'.format(self.__class__.__name__,
                                           self.in_channels, self.out_channels,
                                           self.dim)
