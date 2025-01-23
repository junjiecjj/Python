
# https://blog.csdn.net/2201_75412083/article/details/139999522


import numpy as np
import copy
from matplotlib import pyplot as plt
from scipy.stats import norm
from numpy import linalg as LA
from scipy.optimize import minimize_scalar

def lasso_coordinate_descent(X, y, lmbda, max_iter=1000, tol=1e-5):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)  # 初始化回归系数

    for iteration in range(max_iter):
        for j in range(n_features):  # 对每个特征进行更新
            # 计算当前特征的梯度
            grad = (X[:, j] @ (X @ beta - y)) / n_samples
            # 应用软阈值操作
            beta[j] = max(0, beta[j] - lmbda / n_samples) - (grad - lmbda * np.sign(beta[j]) * (beta[j] > 0))

        # 检查收敛性
        if np.linalg.norm(X @ beta - y, ord=2) < tol:
            break
    return beta

# 近端梯度下降（Proximal Gradient Descent）
def proximal_operator(z, lambda_, L):
    return np.sign(z) * np.maximum(np.abs(z) - lambda_ / L, 0)

def proximal_gradient_descent(X, y, lambda_, L = 1.0, tol = 1e-4, max_iter = 1000):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    for k in range(max_iter):
        # 计算梯度
        grad = (1 / n_samples) * np.dot(X.T, X @ beta - y)
        # 进行近端操作
        z = beta - L * grad
        beta = proximal_operator(z, lambda_ / L, L)
        # 检查收敛性
        if np.linalg.norm(beta - proximal_operator(beta, lambda_ / L, L), ord=2) < tol:
            break
    return beta

def lasso_penalty_method(X, y, lambda_, rho, max_iter = 1000, tol = 1e-5):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    for _ in range(max_iter):
        # 计算当前解的梯度
        grad = np.dot(X.T, X).dot(beta) - np.dot(X.T, y)
        # 应用罚函数
        penalty = lambda_ / rho * np.sign(beta)
        beta -= (grad + penalty) / (np.dot(X.T, X) + rho / n_samples)

        # 检查收敛性
        if np.linalg.norm(grad + penalty, ord=2) < tol:
            break
    return beta

def fista(A, b, lambda_, x0, L0 = 1.05, eta = 1.01, tol = 1e-5, max_iter = 1000):
    m, n = A.shape
    x = x0.copy()
    y = x0.copy()
    t = 1
    g = A.T @ (A @ x - b)
    L = L0

    for k in range(max_iter):
        z = y - (1/L) * g  # 计算近似问题的解
        x = np.sign(z) * np.maximum(np.abs(z) - (lambda_ / L), 0)  # 软阈值操作
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2  # 更新加速因子
        y = x + (t - 1) / t_next * (x - x0)  # FISTA更新公式
        x0 = x.copy()  # 更新x0为当前x
        t = t_next  # 更新加速因子

        # 检查收敛性
        if np.linalg.norm(x - y, 'fro') < tol:
            break

        # 步长更新（使用backtracking line search）
        L = eta * L
        g_new = A.T @ (A @ x - b)
        if np.dot(g_new, g_new - g) < -0.5 * L * np.linalg.norm(g_new - g, 'fro') ** 2:
            L /= eta   # 步长调整
            g = g_new  # 更新梯度
    return x

#%% https://zhuanlan.zhihu.com/p/432897936
# 1. 近端梯度下降法Proximal Gradient Descent
def fx(A,x,b,mu):#lasso的函数值
    f=1/2*np.linalg.norm(A@x-b, ord = 2)**2 + mu*np.linalg.norm(x, ord = 1)
    return f

def Beta(A):#A^\top @A的最大特征值
    return max(np.linalg.eig(A.T@A)[0])
def z(A, x, b):
    beta = Beta(A)
    z = (np.eye(len(x)) - A.T@A/beta) @ x + A.T @ b / beta
    return z
def xp(z, mu, A):#临近点算子
    temp = abs(z) - mu/Beta(A)
    for i in range(len(temp)):
        if temp[i] > 0:
            temp[i] = temp[i]
        else:
            temp[i] = 0
    xp = np.sign(z) * temp
    return xp

def prox(A,x,b,mu,ml):#近端梯度下降算法
    k=0
    fmin=fx(A,x,b,mu)
    fk=fmin
    f_list=[fk]
    while k<ml:
        k=k+1
        x=xp(z(A,x,b),mu,A)
        fk=fx(A,x,b,mu)
        f_list.append(fk)
        if fk<fmin:
            fmin=fk
    plt.scatter(list(range(len(f_list))),f_list,s=5)
    plt.show()
    print("最终迭代结果为：",fmin)


np.random.seed(2021) # set a constant seed to get same random matrixs
A = np.random.rand(500, 100)

x_ = np.zeros([100, 1])
x_[:5, 0] += np.array([i+1 for i in range(5)]) # x_denotes expected x
b = np.matmul(A, x_) + np.random.randn(500, 1) * 0.1 #add a noise to b
lam = 10 # try some different values in {0.1, 1, 10}

prox(A,x_,b,lam,100)
prox(A,x_,b,1,1000)
prox(A,x_,b,0.1,1000)

# 2. BCD块坐标下降法
def BCD(A,x,b,mu):
    k=0
    y=np.ones([100, 1])
    fk=fx(A,x,b,mu)
    f_list=[fk]
    while k<100:#迭代次数，也可以改为其他终止条件
        y=x
        k=k+1
        for i in range(len(x)):
            if x[i][0]>0:
                x[i][0]=1/(A[:,i].T@A[:,i])*(A[:,i].T@b2(A,x,b,i)-mu)
            elif x[i][0]<0:
                x[i][0]=1/(A[:,i].T@A[:,i])*(A[:,i].T@b2(A,x,b,i)+mu)
            elif abs(A[:,i].T@b2(A,x,b,i))<=mu:
                x[i][0]=0
        fk=fx(A,x,b,mu)
        f_list.append(fk)
    plt.scatter(list(range(len(f_list))),f_list,s=5)
    plt.show()
    print("最终迭代结果为：",fx(A,x,b,mu))
def b2(A,x,b,n):
    sum=np.zeros((500,1))
    for i in range(n):
        sum=sum+x[i][0]*A[:,i]
    for i in range(n+1,len(x)):
        sum=sum+x[i][0]*A[:,i]
    b2=b-sum
    return b2

A=np.matrix(A)
BCD(A,x_,b,10)
BCD(A,x_,b,1)
BCD(A,x_,b,0.1)

# 3.ADMM交替方向乘子法
def fxz(A,x,z,b,lam):
    f = 1/2*np.linalg.norm(A@x - b, ord = 2)**2 + lam*np.linalg.norm(z, ord = 1)
    return f
def Beta(A):
    return max(np.linalg.eig(A.T@A)[0])
def xp(z, lam, A):
    temp = abs(z) - lam/Beta(A)
    for i in range(len(temp)):
        if temp[i] > 0:
            temp[i] = temp[i]
        else:
            temp[i] = 0
    xp = np.sign(z)*temp
    return xp

def ADMM(A, x, b, lam):
    mu= np.ones([100, 1])
    rho = Beta(A)
    rho_i = np.identity(A.shape[1])*rho
    z = x
    k = 0
    F = []
    f = fxz(A, x, z, b, lam)
    while k < 100:
        x = np.linalg.inv(A.T@A + rho_i) @ (A.T @ b + rho * (z - mu))
        z = xp(x+mu, lam, A)
        mu = mu + x - z
        k = k + 1
        deltaf = (f - fxz(A, x, z, b, lam))/fxz(A, x, z, b, lam)
        f = fxz(A, x, z, b, lam)
        F.append(f)
    plt.scatter(list(range(0, 100)), F, s = 5)
    plt.show()
    print(fxz(A, x, z, b, lam))

ADMM(A, x_, b, 10)
# ADMM(A, x_, b, 1)
# # ADMM(A, x_, b, 0.1)




#%%

# https://github.com/mihirchakradeo/admm/blob/master/admm/lasso.py

import numpy as np

def coordinate_descent(z, x, nu, rho, lamb):
    e_grad = x + nu*1.0/rho
    rho = 1
    # Regularization term gradient
    # This will have a subgradient, with values as -lambda/rho, lambda/rho OR 0

    # print("prev",z)
    z_t = np.zeros_like(z)

    filter_less = -(1.0*lamb/rho)*(z<0)
    # print("less",filter_less)
    filter_greater = (1.0*lamb/rho)*(z>0)
    # print("gt",filter_greater)

    z_t = e_grad - filter_less - filter_greater
    # print(z_t)
    return(z_t)

d = 20
n = 100

A = np.random.randn(n, d)
b = np.random.randn(n, 1)

X_t = np.random.randn(d, 1)
# z_t = np.random.randn(d, 1)
# X_t = np.zeros((d,1))
z_t = np.zeros((d,1))

rho = 1
# nu_t = np.random.randn(d, 1)
nu_t = np.zeros((d,1))

num_iterations = 10

print(A.shape,b.shape,X_t.shape,z_t.shape,rho,nu_t.shape)
# Initializations
lamb = 0.1
val = 0.5*np.linalg.norm(A.dot(X_t) - b, ord='fro')**2 + lamb*np.linalg.norm(X_t, ord=1)
print(val)
for iter in range(num_iterations):

    ## STEP 1: Calculate X_t
    ## This has a closed form solution
    term1 = np.linalg.inv(A.T.dot(A) + rho)
    term2 = A.T.dot(b) + rho*z_t -  nu_t
    X_t = term1.dot(term2)
    ## print(term1.shape, term2.shape, X_t.shape)

    ## STEP 2: Calculate z_t
    ## Taking the prox, we get the lasso problem again, so, using coordinate_descent
    lamb = 0.1
    z_t = coordinate_descent(z_t, X_t, nu_t, rho, lamb)

    # STEP 3: Update nu_t
    nu_t = nu_t + rho*(X_t - z_t)
    val = 0.5*np.linalg.norm(A.dot(X_t) - b, ord='fro')**2 + lamb*np.linalg.norm(X_t, ord=1)
    print(val)

val = 0.5*np.linalg.norm(A.dot(X_t) - b, ord='fro')**2 + lamb*np.linalg.norm(X_t, ord=1)
print(val)


#%% https://mp.weixin.qq.com/s?__biz=MzI3MzkyMzE5Mw==&mid=2247484248&idx=1&sn=ecc4277e80bc3f355b45668450409490&chksm=eaba5ea6cacd7a7cdb6359b0c305c6c7885e9e2f65d7d3a6bd31788940fa6da8a97c6af36b7e&mpshare=1&scene=1&srcid=0108Dhn6Cvnt00VS8FV3Nop7&sharer_shareinfo=a7ea0763f7e5a219e101eb842bc47515&sharer_shareinfo_first=a7ea0763f7e5a219e101eb842bc47515&exportkey=n_ChQIAhIQ7rnC6DLsAG%2B%2BhSr35iwCyhKfAgIE97dBBAEAAAAAAEi1MYcGxd0AAAAOpnltbLcz9gKNyK89dVj076HLyUbfG%2B%2FDckAZ0PbcEVgNuIYeuPiqYLMSUkKf5dGUT4nP%2BDLoPgGGqhUhnhEApcDnK%2Fj7%2FB8hEqrSBHhvNprpxdJBIZ3f%2FiYfcJ85DI2VgFJ6ew3gLVHsE7eCuamPL%2FzetyIZC0W%2BOPesFP0asly0i4mxQXy0mTrJMigkviOdpoBOesffD36oZumpv26uXNEoIRwUi25gYQPLKVG%2BhJntA%2FIpyj7OxL6TpgaaoQWrMMlPG9IJwNKrDIiwe1OzmG5D%2Bnqs%2BDKllIUxHmWbvhe%2B0t5sqGZfT4ugB7fO6XUB8lddZBG8BWJjDSWAjRQdLgxVezyMphd8&acctmode=0&pass_ticket=HJ9IHHr8aRBHw9nQIKWFfqbctQ6wbw9mUFlFHCaoOwp7odY%2BlEB9CoambnP7%2BJfG&wx_header=0#rd


#### 邻近点梯度下降法

import numpy as np
import random
ASize = (50, 100)
XSize = 100
A = np.random.normal(0, 1, ASize)
X = np.zeros(XSize)
e = np.random.normal(0, 0.1, 50)
XIndex = random.sample(list(range(XSize)), 5)  ## 5 稀疏度
for xi in XIndex:
    X[xi] = np.random.randn()

b = np.dot(A, X) + e

ASize = (50, 100)
BSize = 50
XSize = 100
alpha = 0.001
P_half = 0.01
Xk = np.zeros(XSize)
zero = np.zeros(XSize)
while True:
    Xk_half = Xk - alpha * np.dot(A.T, np.dot(A, Xk) - b)
    ## 软门限算子
    Xk_new = zero.copy()
    for i in range(XSize):
        if Xk_half[i] < - alpha * P_half:
            Xk_new[i] = Xk_half[i] + alpha * P_half
        elif Xk_half[i] > alpha * P_half:
            Xk_new[i] = Xk_half[i] - alpha * P_half
    if np.linalg.norm(Xk_new - Xk, ord = 2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()

print(Xk)
print(X)

#### 邻近点梯度下降法
import matplotlib.pyplot as plt
import numpy as np

ASize = (50, 100)
BSize = 50
XSize = 100
alpha = 0.005
P_half = 0.01
Xk = np.zeros(XSize)
zero = np.zeros(XSize)

X_opt_dst_steps = []
X_dst_steps = []
while True:
    Xk_half = Xk - alpha * np.dot(A.T, np.dot(A, Xk) - b)
    # 软门限算子
    Xk_new = zero.copy()
    for i in range(XSize):
        if Xk_half[i] < - alpha * P_half:
            Xk_new[i] = Xk_half[i] + alpha * P_half
        elif Xk_half[i] > alpha * P_half:
            Xk_new[i] = Xk_half[i] - alpha * P_half
    X_dst_steps.append(np.linalg.norm(Xk_new - X, ord=2))
    X_opt_dst_steps.append(Xk_new)
    if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()

print(Xk)
print(X)

X_opt = X_opt_dst_steps[-1]

for i, data in enumerate(X_opt_dst_steps):
    X_opt_dst_steps[i] = np.linalg.norm(data - X_opt, ord=2)
plt.title("Distance")
plt.plot(X_opt_dst_steps, label='X-opt-distance')
plt.plot(X_dst_steps, label='X-real-distance')
plt.legend()
plt.show()

#### 交替方向乘子法
import matplotlib.pyplot as plt
import numpy as np


ASize = (500, 1000)
XSize = 1000
A = np.random.normal(0, 1, ASize)
X = np.zeros(XSize)
e = np.random.normal(0, 0.1, 500)
XIndex = random.sample(list(range(XSize)), 30)  ## 5 稀疏度
for xi in XIndex:
    X[xi] = np.random.randn()

b = np.dot(A, X) + e

P_half = 0.01
c = 0.005
Xk = np.zeros(XSize)
Zk = np.zeros(XSize)
Vk = np.zeros(XSize)

X_opt_dst_steps = []
X_dst_steps = []

while True:
    Xk_new = np.dot( np.linalg.inv(np.dot(A.T, A) + c * np.eye(XSize, XSize)), c*Zk + Vk + np.dot(A.T, b) )

    # 软门限算子
    Zk_new = np.zeros(XSize)
    for i in range(XSize):
        if Xk_new[i] - Vk[i] / c < - P_half / c:
            Zk_new[i] = Xk_new[i] - Vk[i] / c + P_half / c
        elif Xk_new[i] - Vk[i] / c > P_half / c:
            Zk_new[i] = Xk_new[i] - Vk[i] / c - P_half / c
    Vk_new = Vk + c * (Zk_new - Xk_new)

    # print(np.linalg.norm(Xk_new - Xk, ord=2))
    X_dst_steps.append(np.linalg.norm(Xk_new - X, ord=2))
    X_opt_dst_steps.append(Xk_new)
    if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()
        Zk = Zk_new.copy()
        Vk = Vk_new.copy()

print(Xk)
print(X)

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(X, linefmt = 'k--', markerfmt = 'k^',  label="True X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(Xk, linefmt = 'k--', markerfmt = 'k^',  label="ADMM X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
# plt.close()

X_opt = X_opt_dst_steps[-1]
for i, data in enumerate(X_opt_dst_steps):
    X_opt_dst_steps[i] = np.linalg.norm(data - X_opt, ord=2)
plt.title("Distance")
plt.plot(X_opt_dst_steps, label='X-opt-distance')
plt.plot(X_dst_steps, label='X-real-distance')
plt.legend()
plt.show()


#### 次梯度法
import matplotlib.pyplot as plt
import numpy as np

def g_right(x):
    Xnew = x.copy()
    for i, data in enumerate(x):
        if data == 0:
            Xnew[i] = 2 * np.random.random() - 1
        else:
            Xnew[i] = np.sign(x[i])
    return Xnew

# ASize = (50, 100)
# BSize = 50
# XSize = 100
alpha = 0.001
p_half = 0.001
alphak = alpha
i = 0

g = lambda x: 2 * np.dot(A.T, (np.dot(A, x) - b)) + p_half * g_right(x)
Xk = np.zeros(XSize)
X_opt_dst_steps = []
X_dst_steps = []

while True:
    Xk_new = Xk - alphak * g(Xk)
    alphak = alpha / (i + 1)
    i += 1
    X_dst_steps.append(np.linalg.norm(Xk_new - X, ord=2))
    X_opt_dst_steps.append(Xk_new)
    print(np.linalg.norm(Xk_new - Xk, ord=2))
    if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()

print(Xk)
print(X)

X_opt = X_opt_dst_steps[-1]

for i, data in enumerate(X_opt_dst_steps):
    X_opt_dst_steps[i] = np.linalg.norm(data - X_opt, ord=2)
plt.title("Distance")
plt.plot(X_opt_dst_steps, label='X-opt-distance')
plt.plot(X_dst_steps, label='X-real-distance')
plt.legend()
plt.show()






#%% https://blog.csdn.net/qq_57730943/article/details/136859386

# 对偶上升法
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[100], [-120]])
lambda_ = np.zeros_like(x)

x_v, f_v, G_v, L_v, lambda_v = [], [], [], [], []

t_x = 0.1
t_lambda = 0.1
max_iters = 10000
tol = 10 ** (-5)

for i in range(max_iters):
    x[0] = x[0] - t_x * (4 * (x[0] - 1) - lambda_[0])
    x[1] = x[1] - t_x * (2 * (x[1] + 2) - lambda_[1])
    if(x[0] < 2):
        x[0] = 2

    Grad = np.array([[2 - x[0].item()], [- x[1].item()]])
    lambda_ = np.maximum(lambda_ + t_lambda * Grad, 0)

    f = 2 * (x[0] - 1) ** 2 + (x[1] + 2) ** 2
    L = f + lambda_[0] * (2 - x[0]) + lambda_[1] * (-x[1])

    x_v.append(x.copy())
    G_v.append(Grad)
    f_v.append(f)
    L_v.append(L)

    if(abs(f - L) < tol):
        print(i + 1)
        break
print(x)
plt.figure()
plt.plot(range(2, len(L_v) + 1), np.array(f_v[1:]) - np.array(L_v[1:]), '-b*', linewidth = 2)
plt.show()



# ADMM
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f_x(x, y, rho, u):
    return (x - 1) ** 2 + (rho / 2) * np.linalg.norm(2 * x + 3 * y - 5 + u) ** 2
def f_y(x, y, rho, u):
    return (y - 2) ** 2 + (rho / 2) * np.linalg.norm(2 * x + 3 * y - 5 + u) ** 2

x = 1
y = 1
rho = 1
u = 0
max_iters = 100
for i in range(max_iters):
    # print(x,y)
    resx = minimize_scalar(lambda x: f_x(x, y, rho, u), bounds = (0, 3), method = 'bounded')
    x = resx.x
    resy = minimize_scalar(lambda y: f_y(x, y, rho, u), bounds = (1, 4), method = 'bounded')
    y = resy.x
    u = u + 2 * x + 3 * y - 5
    rho = min(2000, 1.1 * rho)
print(x,y)

















































































































































