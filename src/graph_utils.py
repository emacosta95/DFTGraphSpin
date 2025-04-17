import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.sparse import lil_matrix
from numba import njit

def create_random_coupling_graph(num_nodes, edge_prob=0.5,sigma:float=0.5, seed=None):
    np.random.seed(seed)
    
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.rand() < edge_prob:
                # Sample from N(0,1)
                coupling_ij = np.random.normal(0, sigma)
                graph.add_edge(i, j, weight=coupling_ij)

    return graph

def generate_spin_basis(n_sites):
    bin_states = np.array(list(itertools.product([0, 1], repeat=n_sites)))
    return 2 * bin_states - 1  # Map 0 -> -1, 1 -> +1

@njit
def compute_diagonal_entries(spin_basis, couplings):
    n_states, n_sites = spin_basis.shape
    diag_values = np.zeros(n_states)

    for r in range(n_states):
        value = 0.0
        for i in range(n_sites):
            for j in range(n_sites):
                value += couplings[i, j] * spin_basis[r, i] * spin_basis[r, j]
        diag_values[r] = value
    return diag_values
        
def create_the_hamiltonian_coupling_operator(n_sites: int, couplings: np.ndarray):
    spin_basis = generate_spin_basis(n_sites)
    diag_values = compute_diagonal_entries(spin_basis, couplings)

    hamiltonian_j = lil_matrix((2**n_sites, 2**n_sites))
    for r, val in enumerate(diag_values):
        hamiltonian_j[r, r] = val

    return hamiltonian_j