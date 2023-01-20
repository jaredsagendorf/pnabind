import torch
from torch import Tensor

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
