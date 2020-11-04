# Raktim Mitra (timkartar7879@gmail.com, raktimmi@usc.edu)
# Contributions by Jared Sagendorf

# third-party modules
import numpy as np
import networkx as nx

def meshLabelSmoothness(labels, edge_index, pos=None, method="weighted_vertex"):
    
    if edge_index.shape[0] == 2:
        edge_index = edge_index.T
    edges_i = edge_index[:,0]
    edges_j = edge_index[:,1]
    num_edges = edge_index.shape[0]
    
    if(method == "edge"):
        ''' Number of edges connecting similarly labelled vertices divided by total number of edges'''
        return np.sum(np.equal(labels[edges_i], labels[edges_j])).astype(int)/num_edges
        
    elif(method == "weighted_edge"):
        ''' Same as above, except penalties weighted by inverse of edge length'''

        inv_edge_lengths = np.power(np.sqrt(np.sum(np.square(pos[edges_i] - pos[edges_j]), axis=1)), -1)
        scale_factor = np.power(np.sum(inv_edge_lengths), -1)
        eq_edges = np.equal(labels[edges_i], labels[edges_j]).astype(int)
        
        return 1 - scale_factor*np.sum(np.multiply(inv_edge_lengths, 1 - eq_edges))
   
    elif(method == "vertex"):
        '''For each class calculate 
            (number of vertices with all neighbours from same class / total number of members of that class), 
            return average over all classes'''    
        g = nx.from_edgelist(edge_index) 
        one_ring = np.array([np.array(list(g[i].keys())) for i in range(len(labels))])
        classes = np.sort(np.unique(labels))
        total = 0
        for c in classes:
            Sa = 0
            Vc = np.where(labels == c)[0]
            neighbours_per_vertex = one_ring[Vc]
            in_or_out = [np.all(labels[neighbours_per_vertex[i]] == c) for i in range(len(Vc))]
            Sa_temp = np.sum(in_or_out)
            Sa += Sa_temp
            total += (Sa_temp/len(Vc))
        
        return total/len(labels)
    
    elif(method == "weighted_vertex"):
        '''For each class calculate 
                (number of vertices with all neighbours from same class / class size), 
                return average over all classes weighted by inverse of class size'''
        g = nx.from_edgelist(edge_index)
        one_ring = np.array([np.array(list(g[i].keys())) for i in range(len(labels))])
        classes = np.sort(np.unique(labels))
        wt_total = 0
        denom = 0
        for c in classes:
            Sa = 0
            Vc = np.where(labels == c)[0]
            neighbours_per_vertex = one_ring[Vc]
            in_or_out = [np.all(labels[neighbours_per_vertex[i]] == c) for i in range(len(Vc))]
            Sa_temp = np.sum(in_or_out)
            Sa += Sa_temp
            wt_total += (Sa_temp/len(Vc)**2)
            denom += 1/len(Vc)
        return wt_total/denom
    else:
        raise NotImplementedError
