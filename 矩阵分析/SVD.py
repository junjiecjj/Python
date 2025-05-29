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
import scipy


#%% 验证A的奇异值与A^HA的特征值，以及A^HA特征向量的关系
A = np.array([[1, 2, 3], [4, 5, 6]])
# A = np.random.randn(2, 3) + 1j * np.random.randn(2, 3)
U, s, VH = np.linalg.svd(A)

AHA = A.conj().T@A
lambda_, V_ = np.linalg.eig(AHA)





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

#  验证 AA^H = US^2U^H,  U是AA^H的特征值，也是A的左奇异向量
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
print(f"Uhat = \n{Uhat}") # == U, 顺序可能改变
# array([[-0.92236578,  0.3863177 ],
#        [ 0.3863177 ,  0.92236578]])
print(f"S @ S.conj().T = \n{S @ S.conj().T}")
# S @ S.conj().T =
# [[90.40267253  0.        ]
#  [ 0.          0.59732747]]

scipy.linalg.inv(U) @ A@A.T @ U == S @ S.conj().T # 说明U是X*X^T的特征向量


# 验证 A^HA = VS^2V^H,  V是A^HA的特征值，也是A的右奇异向量.注意，特征值和特征向量的顺序可能改变，且正负号也可能变，但是对应关系不会变
V = VH.conj().T
AHA =  A.conj().T @ A
VS2VH = V @ S.conj().T @ S @  V.conj().T
Lambda2, Vhat = np.linalg.eigh(AHA)

print("V = \n",V)
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
print(f"Vhat = \n{Vhat}") # == V,  顺序可能改变
# array([[ 0.40824829, -0.80596391, -0.42866713],
#        [-0.81649658, -0.11238241, -0.56630692],
#        [ 0.40824829,  0.58119908, -0.7039467 ]])
print(f"S.conj().T @ S = \n{S.conj().T @ S}")
# S.conj().T @ S =
# [[90.40267253  0.          0.        ]
#  [ 0.          0.59732747  0.        ]
#  [ 0.          0.          0.        ]]

scipy.linalg.inv(V) @ A.T@A @ V == S.conj().T @ S # 说明V是X^T*X的特征向量


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

# 验证 AA^H = US^2U^H,  U是AA^H的特征值，也是A的左奇异向量
AAH = A @ A.conj().T
US2UH = U @ S @ S.conj().T @ U.conj().T
Lambda1, Uhat = np.linalg.eigh(AAH) # or use eigh
print(f"AAH = \n{AAH}")

print(f"US2UH = \n{US2UH}")

print(f"Lambda1 = \n{Lambda1}")

print(f"Uhat = \n{Uhat}")

print(f"S @ S.conj().T = \n{S @ S.conj().T}")


#  验证 A^HA = VS^2V^H,  V是A^HA的特征值，也是A的右奇异向量.注意，特征值和特征向量的顺序可能改变，且正负号也可能变，但是对应关系不会变
AHA =  A.conj().T @ A
VS2VH = V @ S.conj().T @ S @  V.conj().T
Lambda2, Vhat = np.linalg.eigh(AHA)

print(f"AHA = \n{AHA}")

print(f"VS2VH = \n{VS2VH}")

print(f"Lambda2 = \n{Lambda2}")

print(f"Vhat = \n{Vhat}")

print(f"S.conj().T @ S = \n{S.conj().T @ S}")


#%% 对原始矩阵 X 进行 SVD 分解, 对中心化数据矩阵 Xc SVD 分解, 对标准化数据矩阵 ZX进行经济型 SVD 分解
# Repeatability
np.random.seed(1)
# Generate random matrix
n = 12
X = np.random.randn(n, 6)

###>>>>>  对原始矩阵 X 进行经济型 SVD 分解
U1, s1, VT1 = np.linalg.svd(X, full_matrices = True)
SS1 = np.zeros(X.shape)
np.fill_diagonal(SS1, s1)
S1 = np.diag(s1)
V1 = VT1.T

G = X.T@X/(n-1)
# 协方差矩阵的特征值分解
Lambda1_, V1_ = np.linalg.eig(G)
Lambda1 = np.diag(Lambda1_)


# X的(奇异值)和 G 的特征值的关系
Lambda1_reproduced = S1**2/(n - 1)
# print(Lambda1_reproduced - Lambda1) # == 0

print(f"s1 = {s1}, Lambda1_ = {Lambda1_}, \nV1 = {V1}")

###>>>>>  对原始矩阵 X 进行经济型 SVD 分解
Xc = X - X.mean(axis =  0)
U2, s2, VT2 = np.linalg.svd(Xc, full_matrices = True)
SS2 = np.zeros(Xc.shape)
np.fill_diagonal(SS2, s2)
S2 = np.diag(s2)
V2 = VT2.T

G2 = Xc.T@Xc/(n-1)
# 协方差矩阵的特征值分解
Lambda2_, V2_ = np.linalg.eig(G2)
Lambda2 = np.diag(Lambda2_)

# Xc的(奇异值)和 G2 的特征值的关系
Lambda2_reproduced = S2**2/(n - 1)
# print(Lambda2_reproduced - Lambda2) # == 0

print(f"s2 = {s2}, Lambda2_ = {Lambda2_}, \nV2 = {V2}")

###>>>>>  对原始矩阵 X 进行经济型 SVD 分解
Zc = Xc @ V2
Zcn = Zc @ np.linalg.inv(np.diag(np.sqrt(Lambda2_)))

U3, s3, VT3 = np.linalg.svd(Zcn, full_matrices = True)
SS3 = np.zeros(Zcn.shape)
np.fill_diagonal(SS3, s3)
S3 = np.diag(s3)
V3 = VT3.T

G3 = Zcn.T@Zcn/(n-1)
# 协方差矩阵的特征值分解
Lambda3_, V3_ = np.linalg.eig(G3)
Lambda3 = np.diag(Lambda3_)


# Zcn 的(奇异值)和 G3 的特征值的关系
Lambda3_reproduced = S3**2/(n - 1)
# print(Lambda3_reproduced - Lambda3) # == 0

print(f"s3 = {s3}, Lambda3_ = {Lambda3_}, \nV3 = {V3}")












































