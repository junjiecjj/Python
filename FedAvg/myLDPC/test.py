#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:32:51 2023

@author: jack
"""

import sys, os
import numpy as np




# enc_H = None

# with open('PEG1024regular0.5.txt', 'r', encoding='utf-8') as f:
#     tmp = f.readline()
#     print(tmp)
#     tmp = f.readline()
#     print(tmp)
#     rows, cols, chk = [int(i) for i in tmp.strip().split()]
#     enc_H = np.zeros((rows, cols), dtype = np.int64)
#     tmp = f.readline()
#     print(tmp)
#     while 1:
#         tmp = f.readline()
#         if not tmp:
#             break
#         row_dt = [int(i) for i in tmp.strip().split()]
#         for i in range(row_dt[1]):
#             enc_H[row_dt[0], row_dt[i+2]] = 1



# orig_decHc_path = '/home/jack/公共的/编解码程序/BinaryLDPC_BPSK_AWGN_simulation_averiter/orig_decH.txt'
# orig_decHp_path = '/home/jack/公共的/Python/FedAvg/myLDPC/orig_decH.txt'

# origc_dec = np.loadtxt(orig_decHc_path, delimiter=' ')
# origp_dec = np.loadtxt(orig_decHp_path, delimiter=' ')
# print((origc_dec - origp_dec).min(),(origc_dec - origp_dec).max())



# decHc_path = '/home/jack/公共的/编解码程序/BinaryLDPC_BPSK_AWGN_simulation_averiter/m_decH.txt'
# decHp_path = '/home/jack/公共的/Python/FedAvg/myLDPC/decH.txt'
# decHc = np.loadtxt(decHc_path, delimiter=' ')
# decHp = np.loadtxt(decHp_path, delimiter=' ')
# print((decHc - decHp).min(),(decHc - decHp).max())




# encHc_path = '/home/jack/公共的/编解码程序/BinaryLDPC_BPSK_AWGN_simulation_averiter/m_encH.txt'
# encHp_path = '/home/jack/公共的/Python/FedAvg/myLDPC/encH.txt'
# encHc = np.loadtxt(encHc_path, delimiter=' ')
# encHp = np.loadtxt(encHp_path, delimiter=' ')
# print((encHc - encHp).min(),(encHc - encHp).max())



# path1 = '/home/jack/公共的/MLData/TrashFile/test1.txt'
# ar1 =  np.arange(24).reshape(4,6)

# np.savetxt(path1, ar1, fmt='%.2f', delimiter=',',)#使用默认分割符（空格），保留两位小数

# path2 = '/home/jack/公共的/MLData/TrashFile/test2.txt'
# np.savetxt(path2, ar1,   delimiter=',', fmt='%.18e')


# ar1_load = np.loadtxt(path1, delimiter=',')#指定逗号分割符
# print(f"ar1_load = \n{ar1_load}")
# print(f"ar1_load.dtype = \n{ar1_load.dtype}")




# ar2_load = np.loadtxt(path2, delimiter=',')#指定逗号分割符
# print(f"ar2_load = \n{ar2_load}")
# print(f"ar2_load.dtype = \n{ar2_load.dtype}")



def hh():
    for iter_num in range(1000):
        if iter_num == 500:
            break

    return iter_num



import numpy as np
# from bitstring import BitArray


# complex method
def Gauss_Elimination(encH, num_row, num_col):
    codechk = 0
    col_exchange = np.arange(num_col)
    ##=======================================================
    ##  开始 Gauss 消元，建立系统阵(生成矩阵G )，化简为: [I, P]的形式
    ##=======================================================
    for i in range(num_row):
        flag = 0
        for jj in range(i, num_col):
            for ii in range(i, num_row):
                if encH[ii, jj] != 0:
                    flag = 1
                    # print("0: I am break")
                    break
            if flag == 1:
                codechk += 1
                # print("1: I am break")
                break
        if flag == 0:
            print("I am break")
            break
        else:
            ## 交换 i 行和 ii 行;
            if ii != i:
                # print(f"{i} 行交换")
                for n in range( num_col):
                    temp =  encH[i, n]
                    encH[i, n] = encH[ii, n]
                    encH[ii, n] = temp
            if jj != i:
                print("0: 列交换")
                ## 记录列交换
                temp = col_exchange[i]
                col_exchange[i] = col_exchange[jj]
                col_exchange[jj] = temp

                ## 交换 i 列和 jj 列;
                for m in range(num_row):
                    temp = encH[m, i]
                    encH[m, i] = encH[m, jj]
                    encH[m, jj] = temp
            ## 消去 [I, P] 形式的前半部分 mxm 矩阵的第 i 列主对角线外的其他元素
            for m in range(num_row):
                if m != i and (encH[m, i] == 1):
                    for n in range(num_col):
                        encH[m, n] ^= encH[i, n]
    ##====================== Gauss 消元 end =================================
    return encH, col_exchange

# efficient method for Gauss elimination
def Gauss_Elimination1(encH, num_row, num_col):
    codechk = 0
    col_exchange = np.arange(num_col)
    ##======================================================================
    ##  开始 Gauss 消元，建立系统阵(生成矩阵G )，化简为: [I, P]的形式
    ##======================================================================
    for i in range(num_row):
        # 获取当前对角线位置 [i, i] 右下角元素中的不为0的元素的索引;
        flag = 0
        for jj in range(i, num_col):
            for ii in range(i, num_row):
                if encH[ii, jj] != 0:
                    flag = 1
                    break
            if flag == 1:
                codechk += 1
                break
        if flag == 0:
            print("I am break")
            break
        else:     # 如果右下角有非零元素,则找出第一个非零元素的行和列;
            ## 交换 i 行和 ii 行;
            if ii != i:
                # print(f"{i} 行交换")
                encH[[i, ii], :] = encH[[ii, i], :]
            if jj != i:
                print("1: 列交换")
                ## 记录列交换
                temp = col_exchange[i]
                col_exchange[i] = col_exchange[jj]
                col_exchange[jj] = temp
                ## 交换 i 列和 jj 列;
                encH[:, [i, jj]] = encH[:, [jj, i]]
            ## 消去 [I, P] 形式的前半部分 mxm 矩阵的第 i 列主对角线外的其他元素
            for m in range(num_row):
                if m != i and (encH[m, i] == 1):
                    # encH[m, :] = encH[m, :] ^ encH[i, :]
                    encH[m, :] = np.logical_xor(encH[m, :], encH[i, :])
                    # encH[m, :] = np.bitwise_xor(encH[m, :], encH[i, :])
                    # for n in range(num_col):
                        # encH[m, n] ^= encH[i, n]
    ##====================== Gauss 消元 end =================================
    return encH, col_exchange


row = 512
col = 1024
arr = np.random.randint(low = 0, high = 2, size = (row, col), dtype = np.int8 )
arr = np.array(
       [[1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]], dtype = np.int8)
arr = np.random.randint(low = 0, high = 2, size = (row, col), dtype = np.int8 )


arr1, col_exg1 = Gauss_Elimination(arr.copy(), row, col )
arr2, col_exg2 = Gauss_Elimination1(arr.copy(), row, col )

print( (arr1-arr2).min(), (arr1-arr2).max() )
# print( (arr-arr2).min(), (arr-arr2).max() )




# arr = np.random.randint(low = 0, high = 2, size = (4, 8), dtype = np.int8 )
# print(f"arr = \n{arr}")


# print(f"np.nonzero(arr[3:,3:]) = \n{np.nonzero(arr[3:,3:])}")



# import copy

# def cange(Arr):
#     Arr[1,1] = 92
#     return Arr

# arr = np.random.randint(low = 0, high = 2, size = (3, 4), dtype = np.int8 )
# print(arr)
# cange(copy.deepcopy(arr))


# print(arr)


























































































































