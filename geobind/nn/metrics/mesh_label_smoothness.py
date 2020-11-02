# Raktim Mitra (timkartar7879@gmail.com, raktimmi@usc.edu)

# third-party modules
import numpy as np
import trimesh   # not sure if this import is required
import networkx as nx

def meshLabelSmoothness(mesh, attribute, method="weighted_vertex"): # requires: trimesh format mesh and mesh attribute
    if(method == "edge"):
        ''' Number of edges connecting similarly labelled vertices divided by total number of edges'''
        return np.sum(np.equal(mesh.vertex_attributes[attribute][mesh.edges[:,0]],
                     mesh.vertex_attributes[attribute][mesh.edges[:,1]]).astype(int))/mesh.edges.shape[0]
        
    elif(method == "weighted_edge"):
        ''' Same as above, except penalties weighted by inverse of edge length'''
        edges_i = mesh.edges[:,0]
        edges_j = mesh.edges[:,1]
        inv_edge_lengths = np.power(np.sqrt(np.sum(np.square(mesh.vertices[edges_i] - mesh.vertices[edges_j]),axis=1)),-1)
        scale_factor = np.power(np.sum(inv_edge_lengths),-1)
        return 1 - scale_factor* np.sum(np.multiply(inv_edge_lengths, 1 - np.equal(mesh.vertex_attributes[attribute][mesh.edges[:,0]],
            mesh.vertex_attributes[attribute][mesh.edges[:,1]]).astype(int)))
   
    elif(method == "vertex"):
        '''For each class calculate 
            (number of vertices with all neighbours from same class / total number of members of that class), 
            return average over all classes'''    
        g = nx.from_edgelist(mesh.edges_unique) 
        one_ring = np.array([np.array(list(g[i].keys())) for i in range(len(mesh.vertices))])
        labels = np.sort(np.unique(mesh.vertex_attributes[attribute]))
        total = 0
        for c in labels:
            Sa = 0
            Vc = np.where(mesh.vertex_attributes[attribute] == c)[0]
            neighbours_per_vertex = one_ring[Vc]
            in_or_out = [np.all(mesh.vertex_attributes[attribute][neighbours_per_vertex[i]] == c) for i in range(len(Vc))]
            Sa_temp = np.sum(in_or_out)
            Sa += Sa_temp
            total += (Sa_temp/len(Vc))
        return total/len(labels)
    
    elif(method == "weighted_vertex"):
        '''For each class calculate 
                (number of vertices with all neighbours from same class / class size), 
                return average over all classes weighted by inverse of class size'''
        g = nx.from_edgelist(mesh.edges_unique) 
        one_ring = np.array([np.array(list(g[i].keys())) for i in range(len(mesh.vertices))])
        labels = np.sort(np.unique(mesh.vertex_attributes[attribute]))
        wt_total = 0
        denom = 0
        for c in labels:
            Sa = 0
            Vc = np.where(mesh.vertex_attributes[attribute] == c)[0]
            neighbours_per_vertex = one_ring[Vc]
            in_or_out = [np.all(mesh.vertex_attributes[attribute][neighbours_per_vertex[i]] == c) for i in range(len(Vc))]
            Sa_temp = np.sum(in_or_out)
            Sa += Sa_temp
            wt_total += (Sa_temp/len(Vc)**2)
            denom += 1/len(Vc)
        return wt_total/denom
    else:
        raise NotImplementedError
