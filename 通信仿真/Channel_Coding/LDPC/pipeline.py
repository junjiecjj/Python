






import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import copy
import torch


def Quantilize(params,  B = 8):
    print(f"B = {B}")
    G =  2**(B - 1)
    params = torch.clamp(torch.round(params * G), min = -1*G, max = G - 1, )/G
    return params

#=======================================================================================

A = torch.randn(5, 3) # .to("cuda")  #torch.rand(size, generator, names)
print(f"A = \n{A}")

Bits = 8

##======================= 1 ============================
G = 2**(Bits - 1)

A1 = A * G
A2 = torch.round(A1)
A3 = torch.clamp(A2, min = -G, max = G - 1, )
A_recv = A3/G
print(f"A_recv = \n{A_recv}")

##======================= 2 ============================

A_recv1 = Quantilize(A.clone(), B = Bits)
print(f"A_recv, {Bits} bit \n{A_recv1}")


print(f"A = \n{A}")


#=======================================================================================
# print(np.iinfo(np.int32).min)
# print(np.iinfo(np.int32).max)
# # -2147483648
# # 2147483647

# print(np.iinfo(np.uint32).min)
# print(np.iinfo(np.uint32).max)
# # 0
# # 4294967295

import torch
data = torch.randn(5, 8)


def Quantilize_end2end(params,  B = 8):
    print(f"B = {B}")
    G =  2**(B - 1)
    params = torch.clamp(torch.round(params * G), min = -1*G, max = G - 1, )/G
    return params




def Quantilize(params,  B = 8):
    print(f"{B} Bit quantization")
    G =  2**(B - 1)

    params = torch.clamp(torch.round(params * G), min = -1*G, max = G - 1, )
    return params  # .type(torch.int8)


print(f"data = \n{data}")

# print(f"A = \n{A}")
num_bits = 8
G = 2**(num_bits - 1)

data_end2end = Quantilize_end2end(data, B = num_bits)
print(f"data_end2end = \n{data_end2end}")

data_qtized = Quantilize(data, B = num_bits)
# print(f"A_recv, \n{A_recv1}")

data_qtized_f = data_qtized.flatten()

dt_up = data_qtized_f + G
# dt_up_np = np.clip(np.array(dt_up, dtype = np.uint64 ), -2**(num_bits-1), 2**(num_bits-1) - 1 ).astype(np.uint64)
dt_up_np =  np.array(dt_up, dtype = np.uint32 )

len_bits = num_bits

bin_len = int(dt_up_np.size * num_bits)
res = np.zeros(bin_len, dtype = np.int8 )
send = np.zeros(bin_len, dtype = np.int8 )


for idx, num in enumerate(dt_up_np):
    res[idx*num_bits:(idx+1)*num_bits] = [int(b) for b in  np.binary_repr(num, width = num_bits+1)[-num_bits:]]

for idx, num in enumerate(dt_up_np):
    send[idx*num_bits:(idx+1)*num_bits] = [int(b) for b in bin(num)[2:].zfill(num_bits)]

##================================= recv =====================================================================

recv = res.copy()
num_dig = int(recv.size/num_bits)

recover = np.zeros(num_dig, dtype = np.uint32 )

for idx in range(num_dig):
    recover[idx] = int(''.join([str(num) for num in recv[idx*num_bits:(idx+1)*num_bits]]), 2)


data_end = (recover*1.0 - G)/G

data_end1 = data_end.reshape(data.shape)

##============================================================================================================
##============================================================================================================

def Quantilize_end2end1(params,  B = 8):
    print(f"B = {B}")
    G =  2**(B - 1)
    params = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1, )/G
    return params

def Quantilize_np(params,  B = 8):
    print(f"{B} Bit quantization")
    G =  2**(B - 1)
    params = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1, )
    return params  # .type(torch.int8)

# print(f"A = \n{A}")
num_bits = 16
G = 2**(num_bits - 1)
data_np = np.array(data)

print(f"data = \n{data}")
datanp_end2end = Quantilize_end2end1(data_np, B = num_bits)
print(f"data_end2end = \n{data_end2end}")


datanp_qtized = Quantilize_np(data_np, B = num_bits)
# print(f"A_recv, \n{A_recv1}")

datanp_qtized_f = datanp_qtized.flatten()

dtnp_up = datanp_qtized_f + G
# dt_up_np = np.clip(np.array(dt_up, dtype = np.uint64 ), -2**(num_bits-1), 2**(num_bits-1) - 1 ).astype(np.uint64)
dtnp_up_np =  np.array(dtnp_up, dtype = np.uint32 )

len_bits = num_bits

bin_len = int(dtnp_up_np.size * num_bits)
resnp = np.zeros(bin_len, dtype = np.int8 )
sendnp = np.zeros(bin_len, dtype = np.int8 )


for idx, num in enumerate(dtnp_up_np):
    resnp[idx*num_bits:(idx+1)*num_bits] = [int(b) for b in  np.binary_repr(num, width = num_bits+1)[-num_bits:]]

for idx, num in enumerate(dtnp_up_np):
    sendnp[idx*num_bits:(idx+1)*num_bits] = [int(b) for b in bin(num)[2:].zfill(num_bits)]

##================================= recv =====================================================================

recvnp = resnp.copy()
num_dig = int(recvnp.size/num_bits)

recovernp = np.zeros(num_dig, dtype = np.uint32 )

for idx in range(num_dig):
    recovernp[idx] = int(''.join([str(num) for num in recvnp[idx*num_bits:(idx+1)*num_bits]]), 2)


datanp_end = (recovernp*1.0 - G)/G

datanp_end1 = datanp_end.reshape(data.shape)


##===============================================================================================================



# def SeqDec2Bin(seqdec, len_dec, len_bits):
#     bin_len = int(seqdec.size * len_bits)
#     res = np.zeros(bin_len, dtype = np.int8 )




for i in range(8):
    print((-12 >> (8-1-i)) & 1, end = ' ')
# 1 1 1 1 0 1 0 0
np.binary_repr(-12, width = 8)
# '11110100'

#================================================================================

import torch
import numpy as np



param_W =  torch.load("/home/jack/FedAvg_DataResults/results/param.pt")

pam_size_len = {}
pam_order = []
params_float = np.empty((0, 0))
num_sum = 0

for key, val in param_W.items():
    pam_order.append(key)
    tmp = {}
    tmp_list = []
    tmp_list.append(val.shape)
    tmp_list.append(val.numel())
    num_sum += val.numel()
    pam_size_len[key] = tmp_list
    params_float = np.append(params_float, np.array(val.cpu().clone()))
    # print(key, val.shape)



##============================================================================================================

##====================
##  params: np.array(dtype = np.float)
##  B     : Number of bits for quantization
def Quantization(params,  B = 8):
    print(f"{B} Bit quantization..")
    G =  2**(B - 1)

    Scale_up = params * G

    Round = np.round(Scale_up)

    Clip = np.clip(Round, a_min = -1*G, a_max = G - 1,)

    Shift = Clip + G

    Uint =  np.array(Shift, dtype = np.uint32 )

    bin_len = int(Uint.size * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Uint):
        binary_send[idx*num_bits:(idx+1)*num_bits] = [int(b) for b in  np.binary_repr(num, width = num_bits+1)[-num_bits:]]

    return binary_send

## encoder- decoder
## ......


binary_recv = 0


##================================= receiver =============================================

# recvnp = resnp.copy()
num_dig = int(binary_recv.size/num_bits)

param_recv = np.zeros(num_dig, dtype = np.uint32 )

for idx in range(num_dig):
    param_recv[idx] = int(''.join([str(num) for num in recvnp[idx*num_bits:(idx+1)*num_bits]]), 2)


param_recv = (param_recv*1.0 - G)/G

param_recover = {}
start = 0
end = 0
for key in pam_order:
    end += pam_size_len[key][1]
    param_recover[key] = torch.tensor(param_recv[start:end].reshape(pam_size_len[key][0]))
    start += pam_size_len[key][1]







# a = np.arange(48).reshape(2,2,3,4)
# b = np.arange(48,57).reshape(3,3)
# c = np.arange(57,63).reshape(2,3)
# param = {'a':a, 'b':b, 'c':c}


# all = np.empty(shape = (0,0))
# for key in param:
#     all = np.append(all, param[key])












































































































































































































