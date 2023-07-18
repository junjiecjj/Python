

import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from pylab import tick_params
import copy
import torch
import torchvision
from torchvision import transforms as transforms


def Quantilize(params, G = None, B = 8):
    if type(B) != int or (G != None and type(G) != int):
        raise ValueError("B 必须是 int, 且 G 不为None时也必须是整数!!!")
    if G == None:
        G =  2**B - 1
    params = torch.clamp(torch.round(params * G), min = -2**(B-1), max = 2**(B-1) - 1, )/G
    return params


A = torch.randn(5, 3).to("cuda")  #torch.rand(size, generator, names)
print(f"A = \n{A}")


B = 8
G = 2**(B-1)

A1 = A * G
A2 = torch.round(A1)
A3 = torch.clamp(A2, min = -2**(B-1), max = 2**(B-1) - 1, )
A_recv = A3/G
print(f"A_recv = \n{A_recv}")

A_recv1 = Quantilize(A, B = 8)
print(f"A_recv, 8 bit  = \n{A_recv1}")



A_recv2 = Quantilize(A, B = 2)
print(f"A_recv, 4 bit  = \n{A_recv2}")






