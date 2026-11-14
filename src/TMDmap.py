import numpy as np
import scipy
from scipy.spatial.distance import cdist

def make_kernel(x, eps, mu):
    '''
    Create the kernel matrix for the TMD method.

    No sparsification is performed (don't try to run it on big datasets lol)
    '''
    x= np.asarray(x, dtype=float)
    N, d= x.shape
    K= ker(x, x, eps) * mu(x)**(1/2) / rho(x, x, eps)[None, :]
    return K


def make_generator(K, eps):
    '''
    Create the generator matrix L = (P-I)/2ε for the TMD method.
    '''
    P= K / K.sum(axis=-1)[:, None]
    I= np.eye(K.shape[0])

    return (P - I) / (eps/2)


def rho(x, X, eps):
    '''
    Evaluates the approximate density ρ_ε at points `x`, 
    based on the data provided in `X`

    X :         (N, d)
    x :         (..., d)
    returns :   (...)
    '''
    x= np.asarray(x, dtype=float)
    X= np.asarray(X, dtype=float)
    n, d= X.shape
    return ker(x, X, eps).sum(-1) / (n * (2 * np.pi * eps)**(d/2))

def ker(x,y, eps):
    '''
    Evaluate the "default" kernel, k(x,y)= exp(-|x-y|^2 / 2ε)
    
    x       : (N_1, ..., N_k,, d)
    y       : (M_1, ..., M_l, d)
    k(x,y)  : (N_1, ..., N_k, M_1, ..., M_l)
    '''
    x= np.asarray(x, dtype=float)
    y= np.asarray(y, dtype=float)
    d= x.shape[-1]
    # cdist computes the pairwise distance matrix, but these need to turn into 2D arrays
    sq= cdist(x.reshape(-1, d), y.reshape(-1, d), 'sqeuclidean')
    # we have to return it to the shape we found it
    return np.exp(-sq / (2 * eps)).reshape(x.shape[:-1] + y.shape[:-1])

def dist_sparsify(x, K, cutoff):
    '''
    Sparsify the L matrix using dn sparsification
    '''

    dists= cdist(x, x, 'sqeuclidean')
    result= K.copy()
    result[dists > cutoff]= 0
    return result

def knn_sparsify(K, k):
    '''
    Delete all but the k largest entries in each row of K
    '''

    result= K.copy()
    idx= np.argpartition(result, -k, axis=-1)[..., :-k]
    np.put_along_axis(result, idx, 0, axis=-1)
    return result