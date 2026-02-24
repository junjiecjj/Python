from os import error, mkdir
import io
import logging
import datetime
import time
from logging import raiseExceptions

# Setup logging
namespace = str(datetime.datetime.now())
figdir = "figures/"+namespace
mkdir(figdir)
logging.basicConfig(filename=figdir+"/log"+namespace+".txt",level=logging.DEBUG,filemode='w')

#from numpy.core.fromnumeric import argmax
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import matplotlib

import pickle

import os
logging.getLogger('PIL').setLevel(logging.WARNING)

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


logging.info("Device is "+ str(device))

if device == 'cuda':
    try:
        import cupy as cp
    except:
        import numpy as cp
else:
    import numpy as cp
#from scipy.stats import ncx2,chi2

time.sleep(10)