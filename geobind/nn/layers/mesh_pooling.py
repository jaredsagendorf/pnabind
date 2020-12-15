# built-in modules
import heapq as hq

# third party modules
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import softmax
from torch_geometric.nn import EdgePooling
from torch_geometric.data import Data

#import trimesh

class Decimator(object):
    @staticmethod
    def computeErrorMetrics(V, F, N):
        # get size of arrays
        nF = F.shape[0]
        nV = V.shape[0]
        
        # need to construct a set of vectors (n, -n*v) for every vertex adjacent to every face
        P = -(N[:,np.newaxis]*V[F]).sum(axis=2) # -n*v for every vertex adjacent to Fi [F, 3]
        P = np.concatenate([
                np.tile(N, (1, 1, 3)).reshape(nF, 3, 3),
                P.reshape(nF, 3, 1)
            ], axis=-1) # [F, 3,  4] holds the vector (n, -n*v) for each vertex in F
        
        # now we take outer product of PP'
        K = np.einsum('ijk,ijl->ijkl', P, P) # this is the error term pp' [F, 3, 4, 4]
        
        # sum over all K for each vertex
        Q = np.zeros((nV, 4, 4))
        np.add.at(Q, F[:,0], K[:,0])
        np.add.at(Q, F[:,1], K[:,1])
        np.add.at(Q, F[:,2], K[:,2])
        
        return Q # [V, 4, 4]
    
    @staticmethod
    def computePairCost(E, Q, V, edge_scores=None, normalize_costs=False, alpha=-1):
        Ne = len(E)
        
        # first we must compute the solution of v(Qi + Qj) = [0,0,0,1]
        Qij = Q[E[:,0]] + Q[E[:,1]] # [E, 4, 4]
        
        vij = np.empty((Ne, 4)) # optimal v for pair vi and vj
        for ei in range(Ne):
            try:
                # attempt to solve for optimal position
                q = Qij[ei].copy()
                q[3,:] = 0
                q[3,3] = 1
                vij[ei] = np.dot(np.linalg.inv(q), _b)
            except:
                # the matrix q is not invertable, fall back to mean of vertex positions
                vij[ei] = (V[E[ei,0]] + V[E[ei,1]])/2
        
        # compute cost vij'*(Qi + Qj)*vij - sij for every pair
        costs = (vij*(Qij*vij[:,np.newaxis]).sum(axis=2)).sum(axis=1)
        
        if normalize_costs:
            costs = (costs - costs.mean())/costs.std()
        
        if edge_scores is not None:
            costs = costs + alpha*edge_scores
        
        return vij, costs
    
    def __init__(self, data, edge_scores, check_manifold=True, normalize_costs=True, alpha=None):
        # Get basic numpy arrays we will need
        self.V = data.pos.cpu().numpy()
        self.F = data.face.cpu().numpy().T
        self.E = data.edge_index.cpu().numpy().copy().T
         
        # Create face normals
        vec1 = self.V[self.F[:,1]] - self.V[self.F[:,0]]
        vec2 = self.V[self.F[:,2]] - self.V[self.F[:,0]]
        face_normals = np.cross(vec1, vec2, axis=1)
        self.N = face_normals/np.linalg.norm(face_normals, axis=1)[:,np.newaxis] # [F, 3]
        
        if edge_scores is not None:
            edge_scores = edge_scores.cpu().detach().numpy()
        
        self.num_vertices = len(self.V)
        self.num_faces = len(self.F)
        self.num_edges = len(self.E)
        
        # Compute quadratic error metrics
        self.Q = self.computeErrorMetrics(self.V, self.F, self.N)
        self.V = np.concatenate([self.V, np.ones((self.num_vertices, 1))], axis=1) # add column of ones
        
        # Compute costs of edges
        self.Vopt, self.edge_costs = self.computePairCost(self.E, self.Q, self.V, edge_scores, normalize_costs=normalize_costs, alpha=alpha)
        
        if check_manifold:
            # construct vertex-vertex adjacency
            vertex_adjacency = [set() for _ in range(self.num_vertices)]
            for i in range(self.num_edges):
                source, target = self.E[i]
                vertex_adjacency[source].add(target)
                vertex_adjacency[target].add(source)
            self.vertex_adjacency = vertex_adjacency
            
            # construct vertex-edge adjacency
            vertex_edges = [{"i": [], "j": []} for _ in range(self.num_vertices)]
            for i in range(self.num_edges):
                source, target = self.E[i]
                vertex_edges[source]["i"].append(i)
                vertex_edges[source]["j"].append(0)
                vertex_edges[target]["i"].append(i)
                vertex_edges[target]["j"].append(1)
            self.vertex_edges = vertex_edges
        
        # Other parameters
        self.check_manifold = check_manifold
        self._vi = self.num_vertices - 1 # vertex index counter
    
    @property
    def vertex_index(self):
        self._vi += 1
        return self._vi
        
    def mergeVertices(self, source, target):
        # replace source and target with a new vertex index
        v = self.vertex_index
        
        # update source edges
        ei = self.vertex_edges[source]["i"]
        ej = self.vertex_edges[source]["j"]
        self.E[ei, ej] = v
        
        # update target edges
        ei = self.vertex_edges[target]["i"]
        ej = self.vertex_edges[target]["j"]
        self.E[ei, ej] = v
        
        # update vertex adjacency
        neighbors_source = self.vertex_adjacency[source]
        neighbors_source.remove(target)
        for n in neighbors_source:
            self.vertex_adjacency[n].remove(source)
            self.vertex_adjacency[n].add(v)
        
        neighbors_target = self.vertex_adjacency[target]
        neighbors_target.remove(source)
        for n in neighbors_target:
            self.vertex_adjacency[n].remove(target)
            self.vertex_adjacency[n].add(v)
        
        self.vertex_adjacency.append(neighbors_source.union(neighbors_target))
    
    def manifoldEdge(self, source, target):
        # check size of the one-ring of the edge [source, target]
        one_ring_intersection = self.vertex_adjacency[source].intersection(self.vertex_adjacency[target])
        
        return len(one_ring_intersection) == 2
    
    def contractEdge(self, ei):
        source, target = self.E[ei]
        # check if this contraction is valid
        #if check_normals and checkNormalsFlip(ei, E, F, N, vopt, V):
            # invalid, return empty list
            #print("triangle flip!", E[ei])
            #mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
            #fmask = (F == E[ei][0]).any(axis=1) + (F == E[ei][1]).any(axis=1)
            #mesh.update_faces(fmask)
            #mesh.export("debug.off")
            #exit(0)
            #return [], []
            #return False
        
        if self.check_manifold and not self.manifoldEdge(source, target):
            # invalid, return empty list
            #return [], []
            #print(self.vertex_adjacency[source])
            #print(self.vertex_adjacency[target])
            #print(one_ring_intersection)
            #print(self.V[list(one_ring_intersection)])
            #print("non-manifold contraction!", self.E[ei])
            #mesh = trimesh.Trimesh(vertices=self.V[:,0:3], faces=self.F, process=False)
            #fmask = (self.F == target).any(axis=1) + (self.F == source).any(axis=1)
            #mesh.update_faces(fmask)
            #mesh.export("debug.off")
            #exit(0)
            return False
        
        # move source and target to optimal position
        self.V[target] = self.Vopt[ei]
        #V[source] = 0.0#vopt[ei]
        
        # update error metric
        #q = Q[source] + Q[target]
        #Q[target] = q
        #Q[source] = q
        
        # update all edges containing source to now point to target
        #emask = (E == source)
        #E[emask[:,0], 0] = target
        #E[emask[:,1], 1] = target
        if self.check_manifold:
            self.mergeVertices(source, target)
        
        #ei = np.argwhere(np.bitwise_or.reduce(emask, axis=1)).flatten()
        #ei, fi = mesh.mergeNodes(source, target)
        
        # update costs and optimal positions of the updated edges containing target
        #vij, costs = MeshPooling.computePairCost(E[ei], scores[ei], Q, V)
        #vopt[ei] = vij
        
        #return ei, costs
        return True

class PriorityQueue(object):
    def __init__(self, data):
        self.heap = [] # list of entries arranged in a heap
        self.entry_finder = {} # mapping of items to entries in heap
        
        # initialize the heap
        for priority, item in data:
            self.addItem(item, priority)
    
    def __len__(self):
        return len(self.heap)
    
    def addItem(self, item, priority=0):
        """Add a new item or update the priority of an existing item"""
        if item in self.entry_finder:
            self.removeItem(item)
        entry = [priority, item, True]
        self.entry_finder[item] = entry
        hq.heappush(self.heap, entry)
    
    def removeItem(self, item):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        if item in self.entry_finder:
            entry = self.entry_finder.pop(item)
            entry[-1] = False
    
    def popItem(self):
        """Remove and return the lowest priority item. Return None if empty."""
        while self.heap:
            priority, item, valid = hq.heappop(self.heap)
            if valid:
                del self.entry_finder[item]
                return item
        
        return None # heap is empty

class MeshPooling(EdgePooling):
    def __init__(self, in_channels, edge_dim,
            dropout=0.0, pre_transform=None, post_transform=None, aggr="diff", alpha=-1.0,
            check_normals=False, check_manifold=False, edge_score_method=None, add_to_edge_score=0.5
        ):
        super().__init__(in_channels,
            add_to_edge_score=add_to_edge_score,
            edge_score_method=edge_score_method,
            dropout=dropout
        )
        
        # parameters
        self.edge_dim = edge_dim
        self.aggr = aggr
        self.check_normals = check_normals
        self.check_manifold = check_manifold
        self.alpha = alpha
        
        # edge feature transforms
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        
        # learnable parameters
        if alpha is not None:
            if aggr == "cat":
                self.lin = torch.nn.Linear(2*in_channels + edge_dim, 1)
            else:
                self.lin = torch.nn.Linear(in_channels + edge_dim, 1)
            self.reset_parameters()
    
    def forward(self, x, data):
        if self.pre_transform:
            data = self.pre_transform(data)
        
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        if self.alpha is not None:
            # compute the edge scores
            if self.aggr == "diff":
                e = torch.cat([torch.abs(x[edge_index[0]] - x[edge_index[1]]), edge_attr], dim=-1)
            elif self.aggr == "mean":
                e = torch.cat([(x[edge_index[0]] + x[edge_index[1]])/2, edge_attr], dim=-1)
            elif self.aggr == "sum":
                e = torch.cat([(x[edge_index[0]] + x[edge_index[1]]), edge_attr], dim=-1)
            elif self.aggr == "max":
                e = torch.cat([torch.maximum(x[edge_index[0]], x[edge_index[1]]), edge_attr], dim=-1)
            elif self.aggr == "cat":
                e = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)
            else:
                raise ValueError("Unsupported option '{}' for aggregation method.".format(self.aggr))
            
            e = self.lin(e).view(-1)
            e = F.dropout(e, p=self.dropout, training=self.training)
            e = self.compute_edge_score(e, edge_index, x.size(0))
            e = e + self.add_to_edge_score
        else:
            e = None
        
        # perform mesh decimation
        new_x, new_data = self.__merge_edges__(x, data, e)
        if self.post_transform:
            new_data = self.post_transform(new_data)
        
        return new_x, new_data
    
    def __merge_edges__(self, x, data, edge_score):
        # Torch tensors
        batch = data.batch
        edge_index = data.edge_index
        
        # Get mesh decimator object
        decimator = Decimator(data, edge_score, check_manifold=self.check_manifold, alpha=self.alpha)

        # Build a priority queue to store edge costs and store which nodes are still valid
        PQ = PriorityQueue([(decimator.edge_costs[i], i) for i in range(decimator.num_edges)])
        
        # Loop over edges, contracting edges and updating node positions
        nodes_remaining = set(range(decimator.num_vertices))
        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        new_edge_indices = []
        i = 0
        while len(PQ) > 0:
            ei = PQ.popItem()
            
            # check if nodes have already been merged
            source, target = decimator.E[ei]
            if (source not in nodes_remaining) or (target not in nodes_remaining):
                continue
            
            contracted = decimator.contractEdge(ei)
            if contracted:
                # this edge was successfully contracted
                nodes_remaining.remove(source)
                cluster[source] = i
                if source != target:
                    nodes_remaining.remove(target)
                    cluster[target] = i
                
                ## update edge costs
                #for j in range(len(updated_edges)):
                    #PQ.addItem(updated_edges[j], updated_costs[j])
                
                i += 1
                new_edge_indices.append(ei)
        
        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)
        
        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        if edge_score is not None:
            new_edge_score = edge_score[new_edge_indices]
            if len(nodes_remaining) > 0:
                remaining_score = x.new_ones(
                    (new_x.size(0) - len(new_edge_indices), )
                )
                new_edge_score = torch.cat([new_edge_score, remaining_score])
            new_x = new_x * new_edge_score.view(-1, 1)
        else:
            new_edge_score = x.new_ones((new_x.size(0), ))
        
        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)
        
        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)
        
        unpool_info = self.unpool_description(
            edge_index=edge_index,
            cluster=cluster,
            batch=batch,
            new_edge_score=new_edge_score
        )
        
        # update mesh vertices
        #vi = torch.empty_like(new_batch, device=torch.device('cpu'))
        #vi[cluster] = torch.arange(cluster.size(0))
        #new_pos = data.pos[vi][:,0:3]
        vi = np.empty(i, dtype=np.int64)
        vi[cluster.cpu().numpy()] = np.arange(decimator.num_vertices)
        new_pos = torch.tensor(decimator.V[vi][:,0:3])
        
        # update faces
        new_face = torch.empty_like(data.face)
        new_face[0,:] = cluster[data.face[0,:]] # assign vertices to their new cluster id
        new_face[1,:] = cluster[data.face[1,:]] # assign vertices to their new cluster id
        new_face[2,:] = cluster[data.face[2,:]] # assign vertices to their new cluster id
        fi = (new_face[0,:] == new_face[1,:]) + (new_face[0,:] == new_face[2,:]) + (new_face[1,:] == new_face[2,:]) # faces with duplicate vertices
        new_face = new_face[:,~fi] # remove duplicates
        
        new_data = Data(edge_index=new_edge_index, batch=new_batch, pos=new_pos, face=new_face)
        new_data.unpool_info = unpool_info
        
        return new_x, new_data
