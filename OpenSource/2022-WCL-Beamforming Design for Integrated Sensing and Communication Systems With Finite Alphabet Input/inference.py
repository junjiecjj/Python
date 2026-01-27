import numpy as np
import h5py
import torch
from model import *
import matplotlib.pyplot as plt
import scipy.io as scio


feature = h5py.File('test_data.mat')
H_data = feature['H_data'][:]
H_data = np.transpose(H_data)

H_data_tensor = torch.from_numpy(H_data)
H_data_tensor = H_data_tensor.to(torch.float32)

H_data_tensor=(H_data_tensor+3.5000)/7

test_data_x =H_data_tensor[0:500]

net = torch.load('mat.pkl')

test_output = net(test_data_x[:500])

test_output = test_output/2 - 0.2500

test_output=test_output.detach().numpy()
scio.savemat('dl.mat', {'dl_data':test_output})

