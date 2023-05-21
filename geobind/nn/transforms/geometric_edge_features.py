# third party modules
import torch
from torch_geometric.utils import to_trimesh 

# gemenai modules
from gemenai.mesh import getGeometricEdgeFeatures

class GeometricEdgeFeatures(object):
    r"""docstring
    """

    def __init__(self, assign_edges=True, pp_features=True, triangle_features=True, n_components=0):
        self.assign_edges=assign_edges
        self.pp_features = pp_features
        self.triangle_features = triangle_features
        self.n_components = n_components
        
        if n_components > 0:
            self.edge_dim = n_components
        else:
            self.edge_dim = 4*pp_features + 5*triangle_features
    
    def __call__(self, data):
        assert data.face is not None
        assert data.pos is not None
        assert data.pos.size(-1) == 3
        assert data.face.size(0) == 3 
        
        mesh = to_trimesh(data)
        edge_index, edge_attr = getGeometricEdgeFeatures(mesh, 
            pp_features=self.pp_features,
            triangle_features=self.triangle_features,
            n_components=self.n_components
        )
        
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
