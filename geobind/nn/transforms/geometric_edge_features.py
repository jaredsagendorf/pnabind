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
        
        data.edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(data.x.device)
        if self.assign_edges:
            data.edge_index = torch.tensor(edge_index.T, dtype=torch.int64).to(data.x.device)
        
        return data
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
