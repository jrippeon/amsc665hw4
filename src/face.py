import numpy as np

rA= 0.5
rB= 0.5
min_A= np.array([-3, 3])
min_B= np.array([0, 4.5])


def V(xy):
    '''
    The face potential

    xy      : (..., 2)
    returns : (...)
    '''
    xy = np.asarray(xy, dtype=float)
    xa=-3
    ya=3
    xb=0
    yb=4.5
    x = xy[..., 0]
    y = xy[..., 1]
    f=(1-x)**2+(y-0.25*x**2)**2+1
    g1=1-np.exp(-0.125*((x-xa)**2+(y-ya)**2))
    g2=1-np.exp(-0.25*(((x-xb)**2+(y-yb)**2)))
    g3=1.2-np.exp(-2*((x+0)**2+(y-2)**2))
    g4=1+np.exp(-2*(x+1.5)**2-(y-3.5)**2-(x+1)*(y-3.5))
    v = f*g1*g2*g3*g4
    return v

def ABint_mask(x):
    '''
    given a number of points x (shape (N, 2)),
    return masks for those in A and B, and the interior.
    '''
    mask_A= np.sum((x - min_A)**2, axis=-1) < rA**2
    mask_B= np.sum((x - min_B)**2, axis=-1) < rB**2
    mask_int= np.ones_like(mask_A, dtype=bool)
    mask_int &= ~mask_A
    mask_int &= ~mask_B
    return mask_A, mask_B, mask_int