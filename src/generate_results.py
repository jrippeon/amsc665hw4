import numpy as np
import matplotlib.pyplot as plt
from face import V, ABint_mask
import TMDmap as tmd
from time import time
from scipy.spatial.distance import cdist
from scipy.differentiate import derivative

from pathlib import Path
ROOT= Path(__file__).resolve().parents[1]

data_dir= ROOT / 'data'



# load data
dd= np.load(data_dir / 'FEM_pts_comm.npz')
pts= dd['pts']
N, d= pts.shape
# dists= cdist(pts, pts).flatten()
# print(f'Minimum nonzero distance is {np.min(dists[dists > 0])}')

q_exact= dd['comm']

mask_A, mask_B, mask_int= ABint_mask(pts)

# parameter values
beta= 3
mu= lambda x: np.exp(-beta * V(x))

# if we set the cutoff too low the kernel just gets obliterated lol
eps_sparsify= lambda K, eps: tmd.dist_sparsify(pts, K, np.max([0.1, 9.0*eps]))
# run_name= 'dn'
run_name= 'knn'


def test_p_values(p_vals, sparsify):


    errors= np.zeros((N_p_vals,N))
    
    build_time= 0.0
    solve_time= 0.0

    for i, p in enumerate(p_vals):
        print(f'{i:02d}: Testing ε=10^{p:0.4f}')

        eps= 10.0**p

        s= time()

        K= tmd.make_kernel(pts, eps, mu)
        K= sparsify(K, eps)

        L= tmd.make_generator(K, eps)

        e= time()
        build_time+= e-s

        Linterior= L[np.ix_(mask_int, mask_int)]
        LB= L[np.ix_(mask_int, mask_B)]

        # this is the same as a matrix-vector product with a vector of ones
        RHS= -LB.sum(axis=-1)

        q= np.zeros_like(q_exact)
        q[mask_B]= 1

        s= time()
        q[mask_int]= np.linalg.solve(Linterior, RHS)
        e= time()
        solve_time += e-s

        err= np.abs(q - q_exact)
        errors[i]= err

    print(f'Total build time was {build_time}s')
    print(f'Total solve time was {solve_time}s')
    MSE= np.mean(errors**2, axis=-1)
    return MSE


def test_K_sum(p_vals, n):
    '''
    Calculate S(ε)= 1/n^2 Σ [K_ε]_ij

    n is the number of data points
    '''
    S= np.zeros_like(p_vals, dtype=float)
    for i, p in enumerate(p_vals):
        print(f'{i:02d}: Testing ε=10^{p:0.4f}')
        eps= 10.0**p
        K= tmd.ker(pts, pts, eps)
        K= sparsify(K, eps)
        S[i]= K.sum() / n**2

    return S

def epsilon_star(eps, S):
    '''
    Compute ε* = argmax_ε (∂ log S / ∂ log ε)
    '''
    d= np.gradient(np.log(S))
    ind= np.argmax(d)
    return eps[ind]





N_p_vals= 30
p_vals= np.linspace(-5, 3, N_p_vals)
# p_vals= np.linspace(-3.5, -1, N_p_vals)
eps= 10.0**p_vals


# # Ksum test

# S= test_K_sum(p_vals, N)
# estar= epsilon_star(eps, S)

# fig, ax= plt.subplots()

# ax.loglog(eps, S)
# ax.set(
#     xlabel=r'$\epsilon$',
#     ylabel=r'$S(\epsilon)$'
# )
# ax.vlines([estar], S.min(), S.max(), color='r')

# fig.savefig(ROOT / 'writeup' / 'figures' / f'SvsEPS{run_name}.pdf')

# # test MSE vs epsilon

# p_vals= np.linspace(-3.5, -1, N_p_vals)
# eps= 10.0**p_vals

# MSE= test_p_values(p_vals)

# fig, ax= plt.subplots()

# ax.loglog(eps, MSE)
# ax.set(
#     xlabel=r'$\epsilon$',
#     ylabel='MSE'
# )
# ax.vlines([estar], MSE.min(), MSE.max(), color='r')

# fig.savefig(ROOT / 'writeup' / 'figures' / f'MSEvsEPS{run_name}.pdf')
# plt.show()

# kNN test
p_values= np.arange(5, 9+1, dtype=int)
k_values= 2**p_values

for k in k_values:
    print(f'----------------------- k={k:04d} -----------------------')

    sparsify= lambda K, eps: tmd.knn_sparsify(K, k)

    p_vals= np.linspace(-3.5, -1, N_p_vals)
    eps= 10.0**p_vals

    MSE= test_p_values(p_vals, sparsify)

    fig, ax= plt.subplots()

    ax.loglog(eps, MSE)
    ax.set(
        xlabel=r'$\epsilon$',
        ylabel='MSE',
        title=f'k={k}'
    )
    # ax.vlines([estar], MSE.min(), MSE.max(), color='r')

    fig.savefig(ROOT / 'writeup' / 'figures' / f'MSEvsEPS{run_name}_k{k}.pdf')
