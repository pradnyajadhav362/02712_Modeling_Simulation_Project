import numpy as np
import networkx as nx
from scipy.sparse import csc_matrix

def make_network(network_type, nnodes, edge_freq, cliq_edge_freq):
    if network_type == 'random':
        G = nx.erdos_renyi_graph(nnodes, edge_freq)
        edges = np.array(list(G.edges()))
        features = np.random.rand(edges.shape[0], 8)
    elif network_type == 'scale_free':
        G = nx.barabasi_albert_graph(nnodes, 3)
        edges = np.array(list(G.edges()))
        features = np.random.rand(edges.shape[0], 8)
    elif network_type == 'small_world':
        G = nx.watts_strogatz_graph(nnodes, 10, 0.1)
        edges = np.array(list(G.edges()))
        features = np.random.rand(edges.shape[0], 8)
    elif network_type == 'cliq_random':
        degrees = [0]*nnodes
        edges = []
        features = [] # (11) cliq1, cliq2, hi_mut_source, hi_mut_target, rand1, rand2, rand3, rand4, rand5, self_loop, intercept
        for i in range(nnodes-1):
            for j in range(i+1,nnodes):
                if ((i<100 and j<100) and np.random.random()<cliq_edge_freq) or np.random.random()<edge_freq:
                    edges.append([i,j])
                    edges.append([j,i])
                    features.append([0,0,0,0,np.random.random(),np.random.random(),0,1])
                    features.append([0,0,0,0,np.random.random(),np.random.random(),0,1])
                    if (i<50 and j<50):
                        features[-2][0] = 1
                        features[-1][0] = 1
                    if (i>=50 and i<100 and j>=50 and j<100):
                        features[-2][1] = 1
                        features[-1][1] = 1
                    if i == nnodes-1:
                        features[-2][2] = 1
                        features[-1][3] = 1
                    if j == nnodes-1:
                        features[-2][3] = 1
                        features[-1][2] = 1
                    degrees[i] += 1
                    degrees[j] += 1
        for i in range(nnodes):
            edges.append([i,i])
            features.append([0,0,0,0,np.random.random(),np.random.random(),1,1])
        edges = np.array(edges)
        features = np.array(features)
    else:
        raise ValueError("Unknown network type")
    return edges, csc_matrix(features)
