class ScaleEdgeFeatures(object):
    r"""docstring
    """

    def __init__(self, method="norm"):
        self.method = method

    def __call__(self, data):
        assert data.edge_attr is not None
        
        if self.method == "clip":
            # scale edge features to lie within [0,1]
            e_mean = data.edge_attr.mean(axis=0)
            e_std = data.edge_attr.std(axis=0)
            e_min = e_mean - 2*e_std
            e_max = e_mean + 2*e_std
            data.edge_attr = torch.clamp((data.edge_attr - e_min)/(e_max - e_min), min=0.0, max=1.0)
        elif self.method == "norm":
            # scale to zero mean and unit variance
            e_mean = data.edge_attr.mean(axis=0)
            e_std = data.edge_attr.std(axis=0)
            data.edge_attr = (data.edge_attr - e_mean)/e_std
        elif self.method is None:
            pass
        else:
            raise ValueError("Unrecognized scaling method: {}".format(self.method))
        
        return data
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
