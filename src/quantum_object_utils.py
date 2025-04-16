from ManyBodySystemQutip.qutip_class import SpinHamiltonian,SpinOperator
import numpy as np
from typing import List,Dict

def get_zz_dictionary(n_sites:int):
    zz_operators={}
    for i in range(n_sites):
        for j in range(i,n_sites):
            zz_operators[i,j]=(SpinOperator(index=[('z',i,'z',j)],coupling=[1],size=n_sites).qutip_op.data.as_scipy())

    return zz_operators

def get_f_operator(n_sites:int):
    f_operators=[]
    for i in range(n_sites):
        f_operators.append(SpinOperator(index=[('x',i)],coupling=[1],size=n_sites).qutip_op.data.as_scipy())
    
    return f_operators

def get_density_f(psi:np.ndarray,f_operators:List):
    energy_h=np.zeros(len(f_operators))
    for i,f in enumerate(f_operators):
        energy_h[i]=psi.conjugate().dot(f.dot(psi))
    return energy_h
        
def get_zz_matrix(zz_operators:Dict,n_sites:int,psi:np.ndarray):
    zz_matrix=np.zeros((n_sites,n_sites))
    for i in range(n_sites):
        for j in range(i,n_sites):
            value=psi.conjugate().dot(zz_operators[i,j].dot(psi))
            zz_matrix[i,j]=value
            if j!=i:
                zz_matrix[j,i]=value
    return zz_matrix