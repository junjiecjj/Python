import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from functools import partial
from .prox_operators import prox_l1, prox_ridge, prox_lp, prox_elastic_net, LMMSE_denoiser
from .base import Denoiser
eps = 1e-16

        
class L1(Denoiser):
    
    def __init__(self, *args, **kwargs):
        super().__init__(prox_l1, *args, **kwargs)

class L2(Denoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(prox_ridge, *args, **kwargs)

class Lp(Denoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(prox_lp, *args, **kwargs)
        
class ElasticNet(Denoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(prox_elastic_net, *args, **kwargs)

class TwoSidedDenoiser():
    def __init__(self):
        raise NotImplementedError()


if __name__ == '__main__':
    
    input_denoiser = L1(Lambda=2)
    
    print(*input_denoiser(np.array([1]),10))
