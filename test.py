
import cvxpy as cp
import numpy as np


d0 = 51
dv = 2                 # user到AP-RIS垂直距离
D0 = 1.0
C0 = -30               # dB
C0 = 10**(C0/10.0)     # 参考距离的路损
sigmaK2 = -80          # dB
sigmaK2 = 10**(sigmaK2/10.0)  # 噪声功率
gamma = 10             # dB
gamma = 10**(gamma/10.0)    #  信干噪比约束10dB
M = 3     # AP天线数量
N = 4    # RIS天线数量


# 路损参数
alpha_AI = 2      #  AP 和 IRS 之间path loss exponent
alpha_Iu = 2.8    # IRS 和 User 之间path loss exponent
alpha_Au = 3.5    # AP 和 User 之间path loss exponent

d = 40
dAu = np.sqrt(d**2 + dv**2)
dIu = np.sqrt((d0-d)**2 + dv**2)
## h_r
Iu_large_fading = C0 * ((dIu/D0)**(-alpha_Iu))
## h_d
Au_large_fading = C0 * ((dAu/D0)**(-alpha_Au))

G = np.sqrt(C0 * ((d0/D0)**(-alpha_AI))) * np.ones((N, M))
hr = np.sqrt(Iu_large_fading) * np.sqrt(1 / (2 * sigmaK2)) * ( np.random.randn(1,N) + 1j * np.random.randn(1,N) )
hd = np.sqrt(Iu_large_fading) * np.sqrt(1 / (2 * sigmaK2)) * ( np.random.randn(1,M) + 1j * np.random.randn(1,M) )

hr = np.array([[-5.52410927+5.11202098j,  2.21161698+3.83710666j,  -6.87020352+8.59078659j, -4.45867517-1.69307665j]])
hd = np.array([[ 9.63447462-5.59241419j,  4.06630045-7.78999435j,  -0.60554468+3.04178728j]])


Phai = np.diag(hr.flatten()) @ G
A = Phai @ (Phai.T.conjugate())
B = Phai @ hd.T.conjugate()
C = hd @ (Phai.T.conjugate())
C = np.append(C, 0).reshape(1, -1)
R = np.concatenate((A, B), axis = 1)
R = np.concatenate((R, C), axis = 0)




## use cvx to solve
V = cp.Variable((N+1, N+1), hermitian = True)
obj = cp.Maximize(cp.real(cp.trace(R@V)) + cp.norm(hd, 2)**2)
# The operator >> denotes matrix inequality.
constraints = [
    0 << V,
    cp.diag(V) == 1,
    ]
prob = cp.Problem(obj, constraints)
prob.solve()

if prob.status == 'optimal':
     # print("optimal")
     low_bound = prob.value
     # print(V.value)
else:
     print("Not optimal")

L = 1000
#%% method 1: 高斯随机化过程
max_F = -1e13
max_v = -1e13
Sigma, U = np.linalg.eig(V.value)
for i in range(L):
    r = np.sqrt(1/2) * ( np.random.randn(N+1, 1) + 1j * np.random.randn(N+1, 1) )
    v = U @ (np.diag(Sigma)**(1/2)) @ r
    # print(f"v^H @ R @ v = {v.T.conjugate() @ R @ v}, max_F = {max_F}")
    if v.T.conjugate() @ R @ v > max_F:
        max_v = v
        max_F = v.T.conjugate() @ R @ v
try:
    optim_v = np.exp(1j * np.angle(max_v/max_v[-1]))
except Exception as e:
    # print(f"V = {V.value}")
    print(f"Sigma = {Sigma}")
    # print(f"U = {U}")
    # print(f"v = {v}")
    print(f"v^H @ R @ v = {v.T.conjugate() @ R @ v}, max_F = {max_F}")

v = v[:-1]

opti = gamma/(np.linalg.norm(v.T.conjugate() @ (np.diag(hr.flatten()) @ G) + hd, ord = 2)**2 )





