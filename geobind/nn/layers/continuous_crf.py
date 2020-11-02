# Raktim Mitra (timkartar7879@gmail.com, raktimmi@usc.edu)

# third-party modules
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContinuousCRF(MessagePassing):
    def __init__(self): 
        super(ContinuousCRF, self).__init__(aggr='add', niter=4, EPS=1e-6)          # "Add" aggregation.
        self.log_alpha = nn.Parameter(torch.rand(1),requires_grad = True)
        self.register_parameter('log_alpha',self.log_alpha)
        self.log_beta = nn.Parameter(torch.rand(1),requires_grad = True)
        self.register_parameter('log_beta',self.log_beta)
        self.logsigmasq = nn.Parameter(torch.rand(1),requires_grad = True)
        self.register_parameter('logsigmasq',self.logsigmasq)
        self.gij = None                                     # temporary memorisation of gij for each protein to avoid repeat computation
                                                            # will be of size (E x 1) when assigned in the first call to propagate 

    def forward(self, x, edge_index):	                    # edge_index has shape [2, E]
        b = x.clone()
        for k in range(niter):
            x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, b=b)
        self.gij = None                                     # set gij None for the next train graph
        return x

    def message(self, x_j, b_i, b_j):
        ''' For each (i,j) \in E, compute gij'''
        if(self.gij == None):                               # size (E x 1)     ## Calculate g_ij only at the start 
            self.gij = torch.exp( torch.nn.functional.cosine_similarity(b_i, b_j, dim=-1)/(torch.exp(self.logsigmasq))).to(device)
        ''' For each edge (i,j) \in E, copmute gij*xj'''
        gijxj = self.gij.view(-1,1)*x_j                     # size (E x 2)      ## compute g_ijxj for each edge (i,j)
        ret = torch.cat((self.gij.view(-1,1),gijxj),dim=1)  # size (E x 3) ## message (gij, gijxj) for each edge (i,j)
        return ret
    
    def update(self, aggr_out, b, x):                       # aggr_out has size (V x 3)
        ''' For each vertex i \in V, aggregated gij over  its neighbour j's, required in the denominator of the update equation'''
        gij_aggregated_over_j = aggr_out[:,0]               # size (V x 1) # sum_{j \in N(i)}g_ij  \forall{i}
        ''' For each vertex i \in V, aggregated gij*xj over  its neighbour j's, required in the neumerator of the update equation'''
        gijxj_aggregated_over_j = aggr_out[:,1:]            # size (V x 2) # sum_{j \in N(i)}g_ijxj \forall{i}
        
        x = (torch.exp(self.log_alpha)*b + torch.exp(self.log_beta)*gijxj_aggregated_over_j)
        x = x/((torch.exp(self.log_alpha) + torch.exp(self.log_beta)*gij_aggregated_over_j).view(-1,1) + EPS)
        return x


