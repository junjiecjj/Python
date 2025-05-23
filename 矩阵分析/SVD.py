#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:01:29 2024

@author: jack
https://blog.csdn.net/forest_LL/article/details/129507243
https://blog.csdn.net/qq_45889056/article/details/128032969

https://zhuanlan.zhihu.com/p/135396870

https://zhuanlan.zhihu.com/p/480389473

https://blog.csdn.net/qfikh/article/details/103994319

"""
import numpy as np


A = np.array([[1,2,3],[4,5,6]])
U, s, VH = np.linalg.svd(A)

print("U = \n",U)
print("s = \n",s)
print("V^H = \n",VH)

print(f"U^H@U = \n{U.T@U}\n")
print(f"U@U^H = \n{U @ U.T}\n")
print(f"VH^H@VH = \n{VH.T@VH}\n")
print(f"VH^H@VH = \n{VH.T@VH}\n")

np.linalg.norm(VH[0,:])  # = 1
np.linalg.norm(VH[1,:])  # = 1
np.linalg.norm(VH[2,:])  # = 1

np.linalg.norm(VH[:,0])  # = 1
np.linalg.norm(VH[:,1])  # = 1
np.linalg.norm(VH[:,2])  # = 1


np.linalg.norm(U[0,:])  # = 1
np.linalg.norm(U[1,:])  # = 1
# np.linalg.norm(U[2,:])  # = 1

np.linalg.norm(U[:,0])  # = 1
np.linalg.norm(U[:,1])  # = 1
# np.linalg.norm(U[:,2])  # = 1

S = np.zeros(A.shape)
np.fill_diagonal(S, s)
# S = np.diag(s)

print(f"U@S@VH = \n{U@S@VH}")
print(f"A = \n{A}")


##%%  验证 AA^H = US^2U^H,  U是AA^H的特征值，也是A的左奇异向量
AAH = A @ A.conj().T
US2UH = U @ S @ S.conj().T @ U.conj().T
Lambda1, Uhat = np.linalg.eig(AAH)
print(f"AAH = {AAH}")


##%%  验证 A^HA = VS^2V^H,  V是A^HA的特征值，也是A的右奇异向量
V = VH.conj().T
AHA =  A.conj().T @ A
VS2VH = V @ S @ S.conj().T @ V.conj().T
Lambda2, Vhat = np.linalg.eig(AHA)
























































