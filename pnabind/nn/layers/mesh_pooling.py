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

# pnabind modules
from pnabind.nn.utils import MLP

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
    def computePairCost(E, Q, V, normalize_costs=False):
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
        
        # compute cost vij'*(Qi + Qj)*vij for every pair
        costs = (vij*(Qij*vij[:,np.newaxis]).sum(axis=2)).sum(axis=1)
        
        if normalize_costs:
            costs = (costs - costs.mean())/costs.std()
        
        return vij, costs
    
    def __init__(self, data, check_manifold=True, check_triangle_flip=False, normalize_costs=False):
        # Get basic numpy arrays we will need
        self.V = data.pos.cpu().numpy()
        self.F = data.face.cpu().numpy().copy().T
        self.E = data.edge_index.cpu().numpy().copy().T
        
        # Create face normals
        vec1 = self.V[self.F[:,1]] - self.V[self.F[:,0]]
        vec2 = self.V[self.F[:,2]] - self.V[self.F[:,0]]
        face_normals = np.cross(vec1, vec2, axis=1)# + 1e-5 # ensure no zero vector
        self.N = face_normals/(np.linalg.norm(face_normals, axis=1)[:,np.newaxis]) # [F, 3]
        
        self.num_vertices = len(self.V)
        self.num_faces = len(self.F)
        self.num_edges = len(self.E)
        
        # Compute quadratic error metrics
        self.Q = self.computeErrorMetrics(self.V, self.F, self.N)
        
        # Compute costs of edges
        self.V = np.concatenate([self.V, np.ones((self.num_vertices, 1))], axis=1) # add column of ones
        self.Vopt, self.edge_costs = self.computePairCost(self.E, self.Q, self.V, normalize_costs=normalize_costs)
        
        if check_manifold:
            # construct vertex-vertex adjacency
            vertex_adjacency = [set() for _ in range(self.num_vertices)]
            for i in range(self.num_edges):
                source, target = self.E[i]
                vertex_adjacency[source].add(target)
                vertex_adjacency[target].add(source)
            self.vertex_adjacency = vertex_adjacency
            
            # construct vertex-edge adjacency
            vertex_edges = [{"ei": [], "ej": []} for _ in range(self.num_vertices)]
            for i in range(self.num_edges):
                source, target = self.E[i]
                vertex_edges[source]["ei"].append(i)
                vertex_edges[source]["ej"].append(0)
                vertex_edges[target]["ei"].append(i)
                vertex_edges[target]["ej"].append(1)
            self.vertex_edges = vertex_edges
        
        if check_triangle_flip:
            # construct vertex-face adjacency
            vertex_faces = [set() for _ in range(self.num_vertices)]
            for i in range(self.num_faces):
                vertex_faces[self.F[i,0]].add(i)
                vertex_faces[self.F[i,1]].add(i)
                vertex_faces[self.F[i,2]].add(i)
            self.vertex_faces = vertex_faces
        
        # Other parameters
        self.check_manifold = check_manifold
        self.check_triangle_flip = check_triangle_flip
    
    def mergeVertices(self, ei):
        # replace source and target with a new vertex index
        # v = self.vertex_index
        # replace min(e) with max(e)
        source = self.E[ei].min()
        target = self.E[ei].max()
        
        if self.check_manifold:
            # update edge source -> target
            ei = self.vertex_edges[source]["ei"]
            ej = self.vertex_edges[source]["ej"]
            self.E[ei, ej] = target
            
            # # update target edges
            # ei = self.vertex_edges[target]["edge_index"]
            # ej = self.vertex_edges[target]["source_or_target"]
            # self.E[ei, ej] = v
            
            # update vertex adjacency
            #print(target, self.vertex_adjacency[target])
            #print(source, self.vertex_adjacency[source])
            neighbors_source = self.vertex_adjacency[source]
            for n in neighbors_source:
                self.vertex_adjacency[n].remove(source)
                self.vertex_adjacency[n].add(target)
            self.vertex_adjacency[target].update(neighbors_source)
            self.vertex_adjacency[target].remove(target)
            #print(target,self.vertex_adjacency[target])
            
            
            #neighbors_target = self.vertex_adjacency[target]
            #neighbors_target.remove(source)
            # for n in neighbors_target:
                # self.vertex_adjacency[n].remove(target)
                # self.vertex_adjacency[n].add(v)
            
            #self.vertex_adjacency.append(neighbors_source.union(neighbors_target))
        
        if self.check_triangle_flip:
            # remove collapsed faces from face adjacency
            si = self.vertex_faces[source].intersection(self.vertex_faces[target]) # shared faces
            for i in si:
                for j in range(3):
                    if i in self.vertex_faces[self.F[i,j]]:
                        self.vertex_faces[self.F[i,j]].remove(i)
            
            # move source->target in face indices
            fi = list(self.vertex_faces[source]) # all faces adjacent to source
            fmask = np.argwhere(self.F[fi] == source)[:,1]
            self.F[fi, fmask] = target
    
    def manifoldEdgeCheck(self, source, target):
        # check size of the one-ring of the edge [source, target]
        one_ring_intersection = self.vertex_adjacency[source].intersection(self.vertex_adjacency[target])
        
        return len(one_ring_intersection) == 2
    
    def triangleFlipCheck(self, ei):
        source, target = self.E[ei]
        
        # move vertices to hypothetical update location
        x_source = self.V[source].copy()
        x_target = self.V[target].copy()
        
        self.V[source] = self.Vopt[ei]
        self.V[target] = self.Vopt[ei]
        
        # get indices of faces which share either vertex but not both
        fi = list(self.vertex_faces[source].symmetric_difference(self.vertex_faces[target]))
        
        # compute normals
        vectors = np.diff(self.V[self.F[fi]][:,:,0:3], axis=1)
        new_N = np.cross(vectors[:,0], vectors[:,1])
        old_N = self.N[fi]
        
        # check if normals changed sign
        if np.any((new_N*old_N).sum(axis=1) <= 0):
            # print(fi)
            # print(self.V[self.F[fi]])
            # print(vectors)
            # print((new_N*old_N).sum(axis=1))
            # print(old_N)
            # print(new_N)
            # put vertex positions back
            self.V[source] = x_source
            self.V[target] = x_target
            return False
        else:
            # update normals
            self.N[fi] = new_N
            return True
    
    
    def contractEdge(self, ei):
        source, target = self.E[ei]
        
        # check if this contraction is valid
        if self.check_manifold and not self.manifoldEdgeCheck(source, target):
            # import trimesh
            # print("non-manifold edge!", self.E[ei])
            # self.V[source] = self.Vopt[ei]
            # self.V[target] = self.Vopt[ei]
            # fi = list(self.vertex_faces[source].union(self.vertex_faces[target]))
            # mesh = trimesh.Trimesh(vertices=self.V, faces=self.F[fi], process=False)
            # #fmask = (self.F == self.E[ei][0]).any(axis=1) + (self.F == self.E[ei][1]).any(axis=1)
            # #mesh.update_faces(fmask)
            # mesh.export("non-manifold.off")
            # exit(0)
            return False
        
        if self.check_triangle_flip and not self.triangleFlipCheck(ei):
            # import trimesh
            # print("triangle flip!", self.E[ei])
            # self.V[source] = self.Vopt[ei]
            # self.V[target] = self.Vopt[ei]
            # fi = list(self.vertex_faces[source].union(self.vertex_faces[target]))
            # mesh = trimesh.Trimesh(vertices=self.V, faces=self.F[fi], process=False)
            # #fmask = (self.F == self.E[ei][0]).any(axis=1) + (self.F == self.E[ei][1]).any(axis=1)
            # #mesh.update_faces(fmask)
            # mesh.export("flipped.off")
            # exit(0)
            return False
        
        if not self.check_triangle_flip:
            # move source and target to optimal position
            self.V[target] = self.Vopt[ei]
            self.V[source] = self.Vopt[ei]
        
        if self.check_manifold or self.check_triangle_flip:
            self.mergeVertices(ei)
        
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
            dropout=0.0,
            pre_transform=None,
            post_transform=None,
            aggr="diff",
            check_triangle_flip=True,
            check_manifold=True,
            edge_score_method=None,
            normalize_qe=False,
            batch_norm=False,
            act=None,
            add_to_edge_score=0.5
        ):
        super().__init__(in_channels,
            add_to_edge_score=add_to_edge_score,
            edge_score_method=edge_score_method,
            dropout=dropout
        )
        
        # parameters
        self.edge_dim = edge_dim
        self.aggr = aggr
        self.check_triangle_flip = check_triangle_flip
        self.check_manifold = check_manifold
        self.normalize_qe = normalize_qe
        
        # edge feature transforms
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        
        # learnable parameters
        if aggr == "cat":
            channels = [2*in_channels + edge_dim, 1]
        else:
            channels = [in_channels + edge_dim, 1]
        self.lin1 = MLP(channels, batch_norm=batch_norm, act=act)
        self.lin2 = torch.nn.Linear(2, 1)
        self.reset_parameters()
    
    def forward(self, x, data):
        if self.pre_transform:
            data = self.pre_transform(data)
        
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        # compute the edge scores
        if self.aggr == "diff":
            ex = torch.cat([torch.abs(x[edge_index[0]] - x[edge_index[1]]), edge_attr], dim=-1)
        elif self.aggr == "mean":
            ex = torch.cat([(x[edge_index[0]] + x[edge_index[1]])/2, edge_attr], dim=-1)
        elif self.aggr == "sum":
            ex = torch.cat([(x[edge_index[0]] + x[edge_index[1]]), edge_attr], dim=-1)
        elif self.aggr == "max":
            ex = torch.cat([torch.maximum(x[edge_index[0]], x[edge_index[1]]), edge_attr], dim=-1)
        elif self.aggr == "cat":
            ex = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)
        else:
            raise ValueError("Unsupported option '{}' for aggregation method.".format(self.aggr))
        
        ex = self.lin1(ex).view(-1, 1)
        ex = F.dropout(ex, p=self.dropout, training=self.training)
        
        # Get quadratic errors
        decimator = Decimator(data, 
                check_manifold=self.check_manifold,
                normalize_costs=self.normalize_qe,
                check_triangle_flip=self.check_triangle_flip
        )
        errors = torch.tensor(decimator.edge_costs, dtype=torch.float32).view(-1, 1).to(ex.device)
        
        # combine
        ex = torch.cat([ex, errors], dim=-1)
        ex = self.lin2(ex).view(-1)
        ex = self.compute_edge_score(ex, edge_index, x.size(0))
        ex = ex + self.add_to_edge_score
        
        # perform mesh decimation
        new_x, new_data = self.__merge_edges__(x, data, ex, decimator)
        
        if self.post_transform:
            new_data = self.post_transform(new_data)
        
        return new_x, new_data
    
    def __merge_edges__(self, x, data, edge_score, decimator):
        # Torch tensors
        batch = data.batch
        edge_index = data.edge_index

        # Build a priority queue to store edge costs and store which nodes are still valid
        PQ = PriorityQueue([(edge_score[i].item(), i) for i in range(len(edge_score))])
        
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
