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

#%% 实数的情况
A = np.array([[1,2,3],[4,5,6]])
# A = np.random.randn(2, 3) + 1j * np.random.randn(2, 3)
U, s, VH = np.linalg.svd(A)

print("U = \n",U)
print("s = \n",s)
print("V^H = \n",VH)

print(f"U^H@U = \n{U.T@U}\n")
print(f"U@U^H = \n{U @ U.T}\n")
print(f"V@V^H = \n{VH.T@VH}\n")
print(f"V^H@V = \n{VH@VH.T}\n")

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

#%%  验证 AA^H = US^2U^H,  U是AA^H的特征值，也是A的左奇异向量
AAH = A @ A.conj().T
US2UH = U @ S @ S.conj().T @ U.conj().T
Lambda1, Uhat = np.linalg.eigh(AAH) # or use eigh
print(f"AAH = \n{AAH}")
# AAH =
# [[14 32]
#  [32 77]]
print(f"US2UH = \n{US2UH}")
# US2UH =
# [[14. 32.]
#  [32. 77.]]
print(f"Lambda1 = \n{Lambda1}")
# Lambda1 =
# [ 0.59732747 90.40267253]
print(f"Uhat = \n{Uhat}")
# array([[-0.92236578,  0.3863177 ],
#        [ 0.3863177 ,  0.92236578]])
print(f"S @ S.conj().T = \n{S @ S.conj().T}")
# S @ S.conj().T =
# [[90.40267253  0.        ]
#  [ 0.          0.59732747]]

#%%  验证 A^HA = VS^2V^H,  V是A^HA的特征值，也是A的右奇异向量.注意，特征值和特征向量的顺序可能改变，且正负号也可能变，但是对应关系不会变
V = VH.conj().T
AHA =  A.conj().T @ A
VS2VH = V @ S.conj().T @ S @  V.conj().T
Lambda2, Vhat = np.linalg.eigh(AHA)

print(f"AHA = \n{AHA}")
# AHA =
# [[17 22 27]
#  [22 29 36]
#  [27 36 45]]
print(f"VS2VH = \n{VS2VH}")
# VS2VH =
# [[17. 22. 27.]
#  [22. 29. 36.]
#  [27. 36. 45.]]
print(f"Lambda2 = \n{Lambda2}")
# Lambda2 =
# [-7.28862329e-15  5.97327474e-01  9.04026725e+01]
print(f"Vhat = \n{Vhat}")
# array([[ 0.40824829, -0.80596391, -0.42866713],
#        [-0.81649658, -0.11238241, -0.56630692],
#        [ 0.40824829,  0.58119908, -0.7039467 ]])
print(f"S.conj().T @ S = \n{S.conj().T @ S}")
# S.conj().T @ S =
# [[90.40267253  0.          0.        ]
#  [ 0.          0.59732747  0.        ]
#  [ 0.          0.          0.        ]]

#%% 复数的情况
# A = np.array([[1,2,3],[4,5,6]])
A = np.random.randn(2, 3) + 1j * np.random.randn(2, 3)
U, s, VH = np.linalg.svd(A)
V = VH.conj().T
print("U = \n",U)
print("s = \n",s)
print("V^H = \n",VH)
print("V = \n", V )

print(f"U^H@U = \n{U.conj().T@U}\n")
print(f"U@U^H = \n{U @ U.conj().T}\n")
print(f"V @ V^H = \n{VH.conj().T@VH}\n")
print(f"V^H@V = \n{VH@VH.conj().T}\n")

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

#%%  验证 AA^H = US^2U^H,  U是AA^H的特征值，也是A的左奇异向量
AAH = A @ A.conj().T
US2UH = U @ S @ S.conj().T @ U.conj().T
Lambda1, Uhat = np.linalg.eigh(AAH) # or use eigh
print(f"AAH = \n{AAH}")

print(f"US2UH = \n{US2UH}")

print(f"Lambda1 = \n{Lambda1}")

print(f"Uhat = \n{Uhat}")

print(f"S @ S.conj().T = \n{S @ S.conj().T}")


#%%  验证 A^HA = VS^2V^H,  V是A^HA的特征值，也是A的右奇异向量.注意，特征值和特征向量的顺序可能改变，且正负号也可能变，但是对应关系不会变

AHA =  A.conj().T @ A
VS2VH = V @ S.conj().T @ S @  V.conj().T
Lambda2, Vhat = np.linalg.eigh(AHA)

print(f"AHA = \n{AHA}")

print(f"VS2VH = \n{VS2VH}")

print(f"Lambda2 = \n{Lambda2}")

print(f"Vhat = \n{Vhat}")

print(f"S.conj().T @ S = \n{S.conj().T @ S}")



















































