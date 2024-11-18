# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:38:10 2021

@author: Parthe
"""

from functools import partial

    
    
class Denoiser():
    
    def __init__(self, denoising_function, *args, **kwargs):
        self.denoising_function = partial(denoising_function, *args, **kwargs)
        self.__dict__.update(**kwargs)
        
    def __call__(self, *args, **kwargs):
        return self.denoising_function(*args, **kwargs)