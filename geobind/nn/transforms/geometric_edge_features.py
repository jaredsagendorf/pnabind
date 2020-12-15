# third party modules
import torch
from torch_geometric.utils import to_trimesh 

# geobind modules
from geobind.mesh import getGeometricEdgeFeatures

class GeometricEdgeFeatures(object):
    r"""docstring
    """

    def __init__(self, assign_edges=True):
        self.assign_edges=assign_edges

    def __call__(self, data):
        assert data.face is not None
        assert data.pos is not None
        assert data.pos.size(-1) == 3
        assert data.face.size(0) == 3 
        
        mesh = to_trimesh(data)
        edge_index, edge_attr = getGeometricEdgeFeatures(mesh)
        
        if data.x is not None:
            device = data.x.device
        else:
            device = data.edge_index.device
        data.edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(device)
        if self.assign_edges:
            data.edge_index = torch.tensor(edge_index.T, dtype=torch.int64).to(device)
        
        return data
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
