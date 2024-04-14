






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


##============================================================================================================
##============================================================================================================
##      将浮点数量化转为int，再转为uint后, 再转为0-1比特串后传输，torch为工具
##============================================================================================================
##============================================================================================================

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
##      将浮点数量化转为int，再转为uint后传输，再转为0-1比特串后传输，np 为工具
##============================================================================================================
##============================================================================================================

data = param_W['conv2.bias'].cpu().clone()

data = torch.randn(5, 8)

data_np = np.array(data)
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
num_bits = 4
G = 2**(num_bits - 1)


print(f"data = \n{data}")
datanp_end2end = Quantilize_end2end1(data_np, B = num_bits)
print(f"datanp_end2end = \n{datanp_end2end}")


datanp_qtized = Quantilize_np(data_np, B = num_bits)
# print(f"A_recv, \n{A_recv1}")

datanp_qtized_f = datanp_qtized.flatten()

dtnp_up = datanp_qtized_f + G
# dt_up_np = np.clip(np.array(dt_up, dtype = np.uint64 ), -2**(num_bits-1), 2**(num_bits-1) - 1 ).astype(np.uint64)
dtnp_up_np =  np.array(dtnp_up, dtype = np.uint32 )


bin_len = int(dtnp_up_np.size * num_bits)
resnp = np.zeros(bin_len, dtype = np.int8 )
sendnp = np.zeros(bin_len, dtype = np.int8 )


for idx, num in enumerate(dtnp_up_np):
    resnp[idx*num_bits : (idx+1)*num_bits] = [int(b) for b in  np.binary_repr(num, width = num_bits+1)[-num_bits:]]

# for idx, num in enumerate(dtnp_up_np):
#     sendnp[idx*num_bits:(idx+1)*num_bits] = [int(b) for b in bin(num)[2:].zfill(num_bits)]

##================================= recv =====================================================================

recvnp = resnp.copy()
num_dig = int(recvnp.size/num_bits)

recovernp = np.zeros(num_dig, dtype = np.uint32 )

for idx in range(num_dig):
    recovernp[idx] = int(''.join([str(num) for num in recvnp[idx*num_bits:(idx+1)*num_bits]]), 2)


datanp_end = (recovernp*1.0 - G)/G

datanp_end1 = datanp_end.reshape(data.shape)


##============================================================================================================
##============================================================================================================
##      将浮点数量化转为int，再转为补码的0-1比特串后传输，np 为工具
##============================================================================================================
##============================================================================================================


data = torch.randn(10, 10)

data_np = np.array(data)


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

def signed_bin2dec(bin_str: str) -> int:
    '''
    函数功能：2进制补码字符串 -> 10进制整数\n
    输入：2进制补码字符串，不可带正负号，前后可加任意个 \\n 和 空格，数字间可加下划线\n
    输出：10进制整数，只保留负号，正号不保留
    '''
    if (bin_str[:2] == '0b'):
        bin_str = bin_str[2:]
    elif (bin_str[0] == '0'):
        return int(bin_str, base = 2)
    elif (bin_str[0] == '1'):
        a = int(bin_str, base = 2) # 此语句可检查输入是否合法
        return a - 2**len(bin_str)


# print(f"A = \n{A}")
num_bits = 4
G = 2**(num_bits - 1)


print(f"data = \n{data}")
data_end2end = Quantilize_end2end1(data_np, B = num_bits)
print(f"data_end2end = \n{data_end2end}")


data_qtized = Quantilize_np(data_np, B = num_bits)
# print(f"A_recv, \n{A_recv1}")

data_qtized_f = data_qtized.flatten()


# dt_up_np = np.clip(np.array(dt_up, dtype = np.uint64 ), -2**(num_bits-1), 2**(num_bits-1) - 1 ).astype(np.uint64)
dtnp_up_np =  np.array(data_qtized_f, dtype = np.int32 )


bin_len = int(dtnp_up_np.size * num_bits)
res = np.zeros(bin_len, dtype = np.int8 )
send = np.zeros(bin_len, dtype = np.int8 )


for idx, num in enumerate(dtnp_up_np):
    res[idx*num_bits : (idx+1)*num_bits] = [int(b) for b in  np.binary_repr(num, width = num_bits)]

# for idx, num in enumerate(dtnp_up_np):
    # send[idx*num_bits:(idx+1)*num_bits] = [int(b) for b in bin(num)[2:].zfill(num_bits)]

##================================= recv =====================================================================



recv = res.copy()
num_dig = int(recv.size/num_bits)

recover = np.zeros(num_dig, dtype = np.int32 )

for idx in range(num_dig):
    recover[idx] = signed_bin2dec(''.join([str(num) for num in recv[idx*num_bits:(idx+1)*num_bits]]))


data_end = (recover*1.0)/G

data_end1 = data_end.reshape(data.shape).astype(dtype = np.float32)

print(f"data_end2end - data_end1 = \n{data_end2end - data_end1}")
##===============================================================================================================



# def SeqDec2Bin(seqdec, len_dec, len_bits):
#     bin_len = int(seqdec.size * len_bits)
#     res = np.zeros(bin_len, dtype = np.int8 )




for i in range(8):
    print((-12 >> (8-1-i)) & 1, end = ' ')
# 1 1 1 1 0 1 0 0
np.binary_repr(-12, width = 8)
# '11110100'


##============================================================================================================
##============================================================================================================
##   将网络的参数转为实数序列，然后反序列化
##============================================================================================================
##============================================================================================================

import torch
import numpy as np



param_W =  torch.load("/home/jack/FedAvg_DataResults/results/param.pt")

pam_size_len = {}
pam_order = []
# params_float = np.empty((0, 0))
params_float = torch.Tensor()
num_sum = 0

for key, val in param_W.items():
    pam_order.append(key)
    tmp_list = []
    tmp_list.append(val.shape)
    tmp_list.append(val.numel())
    num_sum += val.numel()
    pam_size_len[key] = tmp_list
    params_float = torch.cat((params_float, val.detach().cpu().flatten()))


param_recover = {}
start = 0
end = 0
for key in pam_order:
    end += pam_size_len[key][1]
    param_recover[key] =  params_float[start:end].reshape(pam_size_len[key][0])
    start += pam_size_len[key][1]


##============================================================================================================
##============================================================================================================
##   将网络的参数转为实数序列，然后 flipping, 然后反序列化
##============================================================================================================
##============================================================================================================

import torch
import numpy as np



def QuantizationTorch_int(params, G = None, B = 8):
    # print(f"{B} Bit quantization..")
    if G ==None:
        G =  2**(B - 1)

    Clip = torch.clamp(torch.round(params * G), min = -1*G, max = G - 1, )

    Int = Clip.type(torch.int32)

    # Int =  np.array(Clip, dtype = np.int32)

    bin_len = int(Int.numel() * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Int):
        # print(f"{num} {num.item()}")
        binary_send[idx*B:(idx+1)*B] = [int(b) for b in  np.binary_repr(num.item(), width = B)]
    return binary_send

def signed_bin2dec(bin_str: str) -> int:
    '''
    函数功能：2进制补码字符串 -> 10进制整数\n
    输入：2进制补码字符串，不可带正负号，前后可加任意个 \\n 和 空格，数字间可加下划线\n
    输出：10进制整数，只保留负号，正号不保留
    '''
    if (bin_str[:2] == '0b'):
        bin_str = bin_str[2:]
    elif (bin_str[0] == '0'):
        return int(bin_str, base = 2)
    elif (bin_str[0] == '1'):
        a = int(bin_str, base = 2) # 此语句可检查输入是否合法
        return a - 2**len(bin_str)


def deQuantizationNP_int(bin_recv, B = 8):
    G = 2**(B-1)
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.int32 )
    for idx in range(num_dig):
        param_recv[idx] = signed_bin2dec(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]))
    param_recv = (param_recv*1.0 )/G
    return param_recv.astype(np.float32)

data = torch.randn(10, )

binary_send = QuantizationTorch_int(data,  B = 8)
err_rate = 0.2
binary_recv = np.random.binomial(n = 1, p = err_rate, size = binary_send.size  )
binary_recv = binary_recv ^ binary_send

dt = deQuantizationNP_int(binary_recv, B = 8)



##============================================================================================================
##============================================================================================================
##   将网络的参数转为np，然后转为整数、再转为uint、再量化，恢复
##============================================================================================================
##============================================================================================================

import torch
import numpy as np



param_W =  torch.load("/home/jack/FedAvg_DataResults/results/param.pt")

param_np = {}
for key in param_W:
    param_np[key] = np.array(param_W[key].detach().cpu())



pam_size_len = {}
pam_order = []
params_float = np.empty((0, 0))
num_sum = 0

for key, val in param_np.items():
    pam_order.append(key)
    tmp = {}
    tmp_list = []
    tmp_list.append(val.shape)
    tmp_list.append(val.size)
    num_sum += val.size
    pam_size_len[key] = tmp_list
    params_float = np.append(params_float, val)
    # print(key, val.shape)

B = 8

##============================================================================================================

##  params: np.array(dtype = np.float)
##  B     : Number of bits for quantization
def Quantization(params,  B = 8):
    print(f"{B} Bit quantization..")
    G =  2**(B - 1)

    # Scale_up = params * G
    # Round = np.round(Scale_up)
    # Clip = np.clip(Round, a_min = -1*G, a_max = G - 1,)

    Clip = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1, )

    Shift = Clip + G

    Uint =  np.array(Shift, dtype = np.uint32 )

    bin_len = int(Uint.size * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Uint):
        binary_send[idx*B:(idx+1)*B] = [int(b) for b in  np.binary_repr(num, width = B+1)[-B:]]

    return binary_send



binary_send = Quantization(params_float, B = B)

import math
patch_len = int(math.ceil(binary_send.size / 1024)) * 1024 - binary_send.size
binary_send = np.append(binary_send, np.zeros((patch_len, ), dtype = np.int8 ))



## encoder- decoder
## ......


binary_recv = binary_send.copy()


def deQuantization(bin_recv, B = 8):
    G = 2**(B-1)
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.uint32 )
    for idx in range(num_dig):
        param_recv[idx] = int(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]), 2)
    param_recv = (param_recv*1.0 - G)/G
    return param_recv


param_recv = deQuantization(binary_recv[:-patch_len], B = B)

##================================= receiver =============================================

# # recvnp = resnp.copy()
# num_dig = int(binary_recv.size/num_bits)

# param_recv = np.zeros(num_dig, dtype = np.uint32 )

# for idx in range(num_dig):
#     param_recv[idx] = int(''.join([str(num) for num in recvnp[idx*num_bits:(idx+1)*num_bits]]), 2)


# param_recv = (param_recv*1.0 - G)/G

param_recover = {}
start = 0
end = 0
for key in pam_order:
    end += pam_size_len[key][1]
    param_recover[key] =  param_recv[start:end].reshape(pam_size_len[key][0])
    start += pam_size_len[key][1]


for key in param_recover:
    param_recover[key] = torch.tensor(param_recover[key], dtype = torch.float32).to('cuda')



##============================================================================================================
##============================================================================================================
##   将网络的参数转为np，然后转为整数、再转为int、再对补码量化，恢复
##============================================================================================================
##============================================================================================================

import torch
import numpy as np



param_W =  torch.load("/home/jack/FedAvg_DataResults/results/param.pt")

param_np = {}
for key in param_W:
    param_np[key] = np.array(param_W[key].detach().cpu())



pam_size_len = {}
pam_order = []
params_float = np.empty((0, 0))
num_sum = 0

for key, val in param_np.items():
    pam_order.append(key)
    tmp = {}
    tmp_list = []
    tmp_list.append(val.shape)
    tmp_list.append(val.size)
    num_sum += val.size
    pam_size_len[key] = tmp_list
    params_float = np.append(params_float, val)
    # print(key, val.shape)

B = 8

##============================================================================================================

##  params: np.array(dtype = np.float)
##  B     : Number of bits for quantization
def Quantization(params,  B = 8):
    print(f"{B} Bit quantization..")
    G =  2**(B - 1)

    # Scale_up = params * G
    # Round = np.round(Scale_up)
    # Clip = np.clip(Round, a_min = -1*G, a_max = G - 1,)

    Clip = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1,)

    # Shift = Clip

    Uint =  np.array(Clip, dtype = np.int32)

    bin_len = int(Uint.size * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Uint):
        binary_send[idx*B:(idx+1)*B] = [int(b) for b in  np.binary_repr(num, width = B)]

    return binary_send



binary_send1 = Quantization(params_float, B = B)

import math
patch_len = int(math.ceil(binary_send1.size / 1024)) * 1024 - binary_send1.size
binary_send1 = np.append(binary_send1, np.zeros((patch_len, ), dtype = np.int8 ))



## encoder- decoder
## ......


binary_recv1 = binary_send1.copy()


def signed_bin2dec(bin_str: str) -> int:
    '''
    函数功能：2进制补码字符串 -> 10进制整数\n
    输入：2进制补码字符串，不可带正负号，前后可加任意个 \\n 和 空格，数字间可加下划线\n
    输出：10进制整数，只保留负号，正号不保留
    '''
    if (bin_str[:2] == '0b'):
        bin_str = bin_str[2:]
    elif (bin_str[0] == '0'):
        return int(bin_str, base = 2)
    elif (bin_str[0] == '1'):
        a = int(bin_str, base = 2) # 此语句可检查输入是否合法
        return a - 2**len(bin_str)


def deQuantization(bin_recv, B = 8):
    G = 2**(B-1)
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.int32 )
    for idx in range(num_dig):
        param_recv[idx] = signed_bin2dec(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]))
    param_recv = (param_recv*1.0 )/G
    return param_recv


param_recv1 = deQuantization(binary_recv1[:-patch_len], B = B)

##================================= receiver =============================================

# # recvnp = resnp.copy()
# num_dig = int(binary_recv.size/num_bits)

# param_recv = np.zeros(num_dig, dtype = np.uint32 )

# for idx in range(num_dig):
#     param_recv[idx] = int(''.join([str(num) for num in recvnp[idx*num_bits:(idx+1)*num_bits]]), 2)


# param_recv = (param_recv*1.0 - G)/G

param_recover1 = {}
start = 0
end = 0
for key in pam_order:
    end += pam_size_len[key][1]
    param_recover1[key] =  param_recv1[start:end].reshape(pam_size_len[key][0])
    start += pam_size_len[key][1]


for key in param_recover1:
    param_recover1[key] = torch.tensor(param_recover[key], dtype = torch.float32).to('cuda')




# a = np.arange(48).reshape(2,2,3,4)
# b = np.arange(48,57).reshape(3,3)
# c = np.arange(57,63).reshape(2,3)
# param = {'a':a, 'b':b, 'c':c}


# all = np.empty(shape = (0,0))
# for key in param:
#     all = np.append(all, param[key])





##============================================================================================================
##============================================================================================================
##   将 torch array 参数转为实数序列，然后 flipping, 然后反序列化
##============================================================================================================
##============================================================================================================

import torch
import numpy as np

def Quantilize_end2end1(params,  B = 8):
    print(f"1  B = {B}")
    G =  2**(B - 1)
    params = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1, )/G
    return params

def QuantizationNP_int(params,  B = 8):
    print(f"2  B = {B}")
    # print(f"{B} Bit quantization..")
    G =  2**(B - 1)
    # Scale_up = params * G
    # Round = np.round(Scale_up)
    # Clip = np.clip(Round, a_min = -1*G, a_max = G - 1,)
    Clip = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1,)
    # Shift = Clip
    Int =  np.array(Clip, dtype = np.int32)

    bin_len = int(Int.size * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Int):
        binary_send[idx*B:(idx+1)*B] = [int(b) for b in  np.binary_repr(num, width = B)]

    return binary_send


def QuantizationTorch_int(params, G = None, B = 8):
    print(f"3  B = {B}")
    # print(f"{B} Bit quantization..")
    if G ==None:
        G =  2**(B - 1)

    Clip = torch.clamp(torch.round(params * G), min = -1*G, max = G - 1, )

    Int = Clip.type(torch.int32)

    # Int =  np.array(Clip, dtype = np.int32)

    bin_len = int(Int.numel() * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Int):
        # print(f"{num} {num.item()}")
        binary_send[idx*B:(idx+1)*B] = [int(b) for b in  np.binary_repr(num.item(), width = B)]

    return binary_send

def signed_bin2dec(bin_str: str) -> int:
    '''
    函数功能：2进制补码字符串 -> 10进制整数\n
    输入：2进制补码字符串，不可带正负号，前后可加任意个 \\n 和 空格，数字间可加下划线\n
    输出：10进制整数，只保留负号，正号不保留
    '''
    if (bin_str[:2] == '0b'):
        bin_str = bin_str[2:]
    elif (bin_str[0] == '0'):
        return int(bin_str, base = 2)
    elif (bin_str[0] == '1'):
        a = int(bin_str, base = 2) # 此语句可检查输入是否合法
        return a - 2**len(bin_str)


def deQuantizationNP_int(bin_recv, B = 8):
    print(f"4  B = {B}")
    G = 2**(B-1)
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.int32 )
    for idx in range(num_dig):
        param_recv[idx] = signed_bin2dec(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]))
    param_recv = (param_recv*1.0 )/G
    return param_recv.astype(np.float32)



data = torch.randn(100, )


B = 4
##=======
data_end2end_np = Quantilize_end2end1(np.array(data), B = B)


##=======
binary_send = QuantizationTorch_int(data,  B = B)
err_rate = 0
binary_recv = np.random.binomial(n = 1, p = err_rate, size = binary_send.size  )
binary_recv = binary_recv ^ binary_send
dt = deQuantizationNP_int(binary_recv, B = B)


##=======
binary_send1 = QuantizationNP_int(np.array(data),  B = B)
err_rate = 0
binary_recv1 = np.random.binomial(n = 1, p = err_rate, size = binary_send1.size  )
binary_recv1 = binary_recv1 ^ binary_send1
dt1 = deQuantizationNP_int(binary_recv1, B = B)








































































































































































































