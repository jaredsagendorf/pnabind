import torch
from torch_geometric.nn import EdgePooling as EdgePoolingBase
import torch.nn.functional as F

class EdgePooling(EdgePoolingBase):
    def __init__(self, in_channels, edge_dim=None, edge_score_method=None, dropout=0,
                    add_to_edge_score=0.5):
        super().__init__(in_channels, edge_score_method=edge_score_method, 
                    dropout=dropout, add_to_edge_score=add_to_edge_score)
        
        self.edge_dim = edge_dim # dimension of edge features
        if edge_dim is None:
            self.lin = torch.nn.Linear(2*in_channels, 1)
        else:
            self.lin = torch.nn.Linear(2*in_channels + edge_dim, 1)
        
        self.reset_parameters()
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.
            edge_attr (Tensor): The edge features.
        
        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        if self.edge_dim is None:
            torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        else:
            e = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)
        
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score
        
        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info
