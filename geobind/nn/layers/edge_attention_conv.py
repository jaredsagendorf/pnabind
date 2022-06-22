from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax

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
            add_self_loops: bool = False,
            attention_type: str = "softmax",
            average_heads: bool = False,
            **kwargs
        ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.gate_nn = gate_nn
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.average_heads = average_heads
        
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)
        reset(self.gate_nn)
    
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: Tensor,
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
        
        # do message-passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        
        if self.global_nn is not None:
            out = self.global_nn(out)
        
        return out

    def message(self, 
            x_i: Tensor,
            x_j: Tensor,
            edge_attr: Tensor,
            index: Tensor
        ) -> Tensor:
        
        # get attention
        gate = self.gate_nn(edge_attr)
        if gate.dim() == 1:
            gate = gate.view(-1, 1)
        
        if self.attention_type == "softmax":
            gate = softmax(gate, index=index)
        elif self.attention_type == "sigmoid":
            gate = F.sigmoid(gate)
        
        if self.average_heads:
             gate = gate.mean(dim=1).view(-1, 1)
        
        # update message
        if self.local_nn is not None:
            msg = self.local_nn(x_j)
        else:
            msg = x_j
        (E, F), H = msg.size(), gate.size(1)
        
        msg = msg.view(E, F, 1)*gate.view(E, 1, H)
        
        return msg.view(E, F*H)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')
