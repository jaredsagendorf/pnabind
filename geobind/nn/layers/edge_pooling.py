import torch
import torch.nn.functional as F
from torch_geometric.nn import EdgePooling as EdgePoolingBase
from torch_geometric.data import Data

class EdgePooling(EdgePoolingBase):
    def __init__(self, in_channels, 
                edge_dim=None, edge_score_method=None, dropout=0,
                add_to_edge_score=0.5, pre_transform=None, post_transform=None
        ):
        super().__init__(in_channels,
                edge_score_method=edge_score_method, 
                dropout=dropout,
                add_to_edge_score=add_to_edge_score
        )
        
        self.edge_dim = edge_dim # dimension of edge features
        if edge_dim is None:
            self.lin = torch.nn.Linear(2*in_channels, 1)
        else:
            self.lin = torch.nn.Linear(2*in_channels + edge_dim, 1)
        
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        
        self.reset_parameters()
    
    def forward(self, x, data):
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
        if self.pre_transform:
            data = self.pre_transform(data)
        
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        
        # compute edge score
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
        
        # update the node positions and faces
        ind1 = torch.empty_like(batch, device=torch.device('cpu'))
        ind2 = torch.empty_like(batch, device=torch.device('cpu'))
        
        N = unpool_info.cluster.size(0)
        ind1[unpool_info.cluster] = torch.arange(N)
        ind2[unpool_info.cluster.flip((0))] = torch.arange(N-1, -1, -1)
        new_pos = (data.pos[ind1] + data.pos[ind2])/2
        
        new_face = torch.empty_like(data.face)
        new_face[0,:] = unpool_info.cluster[data.face[0,:]]
        new_face[1,:] = unpool_info.cluster[data.face[1,:]]
        new_face[2,:] = unpool_info.cluster[data.face[2,:]]
        fi = (new_face[0,:] == new_face[1,:]) + (new_face[0,:] == new_face[2,:]) + (new_face[1,:] == new_face[2,:])
        new_face = new_face[:,~fi]
        
        new_data = Data(edge_index=edge_index, pos=new_pos, face=new_face, batch=batch)
        new_data.unpool_info = unpool_info
        
        if self.post_transform:
            new_data = self.post_transform(new_data)
            
        return x, new_data
