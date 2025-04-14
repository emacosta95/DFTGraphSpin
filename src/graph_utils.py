import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_random_coupling_graph(num_nodes, edge_prob=0.5, seed=None):
    np.random.seed(seed)
    
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.rand() < edge_prob:
                # Sample from N(0,1)
                coupling_ij = np.random.normal(0, 1)
                graph.add_edge(i, j, weight=coupling_ij)

    return graph
