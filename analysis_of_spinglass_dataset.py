from ManyBodySystemQutip.qutip_class import SpinHamiltonian,SpinOperator
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.graph_utils import create_random_coupling_graph
import numpy as np
from typing import List,Dict
from tqdm import trange
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse.linalg import eigsh as eigsh_cp
from src.quantum_object_utils import get_f_operator,get_density_f,get_zz_dictionary,get_zz_matrix
ndata=10000
n_sites=16
average_coupling=3
n_levels=3

h_hamiltonian=SpinOperator(index=[('x',i) for i in range(n_sites)],coupling=[1]*n_sites,size=n_sites)

zz_dictionary=get_zz_dictionary(n_sites=n_sites)
f_operators=get_f_operator(n_sites=n_sites)

fs=[]
fs_density=[]
zz_matrices=[]
energies=[]
couplingss=[]

for i in trange(ndata):
    
    # there is a way to build up classically this object (it's diagonal!)
    graph=create_random_coupling_graph(num_nodes=n_sites,edge_prob=average_coupling/n_sites)
    couplings = nx.to_numpy_array(graph, weight='weight')
    dict_couplings={}
    index=[]
    values=[]
    for a in range(n_sites):
        for b in range(n_sites):
            if couplings[a,b]!=0:
                dict_couplings[(a,b)]=couplings[a,b]
                index.append(('z',a,'z',b))
                values.append(couplings[a,b])

    j_hamiltonian=SpinOperator(index=index,coupling=values,size=n_sites)
    
    tot_hamiltonian=j_hamiltonian.qutip_op+h_hamiltonian.qutip_op
    tot_hamiltonian_sparse=cp_csr_matrix(tot_hamiltonian.data.as_scipy())
    es,psis=eigsh_cp(tot_hamiltonian_sparse,k=n_levels,which='SA')
    
    zz_matrix=get_zz_matrix(zz_operators=zz_dictionary,n_sites=n_sites,psi=psis[:,0])
    f_density=get_density_f(psi=psis[:,0],f_operators=f_operators)
    
    f_value=np.average(f_density)
    
    print('check=',f_value*n_sites+np.sum(zz_matrix*couplings)-es[0],'\n')
    
    fs.append(f_value)
    fs_density.append(f_density)
    zz_matrices.append(zz_matrix)
    energies.append(es.get())
    couplingss.append(couplings)

fs=np.asarray(fs)
fs_density=np.asarray(fs_density)
zz_matrices=np.asarray(zz_matrices)
energies=np.asarray(energies)
couplingss=np.asarray(couplingss)

np.savez(f'data/datasets/dataset_for_training_ndata_{ndata}_nsites_{n_sites}_test0',fs=fs,zz_matrices=zz_matrices,energies=energies,couplings=couplingss,fs_density=fs_density)
