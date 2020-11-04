# Author: Raktim Mitra (timkartar7879@gmail.com, raktimmi@usc.edu)
# Contributions by Jared Sagendorf

# third-party modules
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class ContinuousCRF(MessagePassing):
    def __init__(self, alpha=None, beta=None, niter=4, eps=1e-6, gij=None): 
        super(ContinuousCRF, self).__init__(aggr='add')
        if alpha is None:
            # treat as a learnable parameter
            self.log_alpha = nn.Parameter(torch.rand(1), requires_grad=True)
        else:
            # use a fixed scalar
            self.log_alpha = torch.log(torch.tensor(alpha, requires_grad=False))
        
        if beta is None:
            self.log_beta = nn.Parameter(torch.rand(1), requires_grad=True)
        else:
            # use a fixed scalar
            self.log_beta = torch.log(torch.tensor(beta, requires_grad=False))
        
        self.niter = niter
        self.eps = eps
        
        # gij is a similarity measure between nodes i and j
        if gij is None:
            # TODO: generalize to allow different gij 'presets'  
            self.gij = lambda *args: torch.exp(nn.functional.cosine_similarity(args[0], args[1], dim=-1)/torch.exp(args[2]))
            self.log_sigmasq = nn.Parameter(torch.rand(1), requires_grad=True)
        else:
            self.gij = gij
        self.batch_gij = None # cache the value of gij to avoid repeated computations during iterations
    
    def forward(self, x, edge_index):                    
        b = x.clone() # original features
        for k in range(self.niter):
            x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, b=b)
        self.batch_gij = None # reset gij for the next batch
        
        return x
    
    def message(self, x_j, b_i, b_j):
        ''' For each (i,j) \in E, compute gij'''
        if self.batch_gij == None:
            # calculate g_ij only at first iteration
            self.batch_gij = self.gij(b_i, b_j, self.log_sigmasq).view(-1, 1) # size (E x 1)
        ''' For each edge (i,j) \in E, copmute gij*xj'''
        
        # compute g_ijxj for each edge (i,j)
        gijxj = self.batch_gij*x_j # size (E x F)      
        
        # message (gij, gijxj) for each edge (i,j)
        return torch.cat([self.batch_gij, gijxj], dim=1) # size (E x F+1)
    
    def update(self, aggr_out, b, x):                      
        # aggr_out has size (V x 3)
        ''' For each vertex i \in V, aggregated gij over  its neighbour j's, required in the denominator of the update equation'''
        # sum_{j \in N(i)}g_ij  \forall{i}
        gij_aggr_j = aggr_out[:,0].view(-1, 1) # size (V x 1) 
        ''' For each vertex i \in V, aggregated gij*xj over  its neighbour j's, required in the neumerator of the update equation'''
        # sum_{j \in N(i)}g_ijxj \forall{i}
        gijxj_aggr_j = aggr_out[:,1:] # size (V x F) 
        
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)

        return (alpha*b + beta*gijxj_aggr_j)/(alpha + beta*gij_aggr_j + self.eps)
