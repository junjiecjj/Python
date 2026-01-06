import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch.nn.functional as F
import torchvision
import numpy as np
from math import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython import display
import torch.utils.data as Data
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.linalg import block_diag
import datetime
from torch.nn.utils import *
from Transformer_model import *

def Hermitian(X):#torch矩阵共轭转置
    X = torch.real(X) - 1j*torch.imag(X)
    return X.transpose(-1,-2)
def kron(a,b):
    #a与b维度为[batch,N],输出应为[batch,N*N]
    batch = a.shape[0]
    a = a.reshape(batch,-1,1)
    b = b.reshape(batch,1,-1)
    c = a @ b
    return c.reshape(batch,-1) #输出的维度a在前，b在后
def kron_add(a,b):
    #a与b维度为[batch,N],输出应为[batch,N*N]
    batch = a.shape[0]
    a = a.reshape(batch,-1,1)
    b = b.reshape(batch,1,-1)
    c = a + b
    return c.reshape(batch,-1) #输出的维度a在前，b在后

