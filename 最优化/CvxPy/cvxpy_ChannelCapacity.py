



# Import packages.
import cvxpy as cp
import numpy as np
import math
from scipy.special import xlogy
#=================================================================
#            通信通道的容量
# https://www.wuzao.com/document/cvxpy/examples/applications/Channel_capacity_BV4.57.html
# https://www.cvxpy.org/examples/applications/Channel_capacity_BV4.57.html
#=================================================================

def channel_capacity(n, m, P, sum_x = 1):
    '''
    通信通道的容量。
    Boyd和Vandenberghe的《凸优化》，练习4.57，第207页

    我们考虑一个通信通道，输入X(t)∈{1,..,n}，输出Y(t)∈{1,...,m}，t=1,2,...。
    输入和输出之间的关系是统计给定的：
    p_(i,j) = ℙ(Y(t)=i|X(t)=j)，i=1,..,m  j=1,...,n

    P矩阵 ∈ ℝ^(m*n) 被称为信道过渡矩阵，并且该信道被称为离散无记忆信道。假设X具有概率分布，表示为x ∈ ℝ^n，即，
    x_j = ℙ(X=j)，j=1,...,n。

    X和Y之间的互信息由下式给出：
    ∑(∑(x_j p_(i,j)log_2(p_(i,j)/∑(x_k p_(i,k)))))

    然后，信道容量C由下式给出：
    C = sup I(X;Y)。
    使用y = Px进行变量变换，得到：
    I(X;Y)=  c^T x - ∑(y_i log_2 y_i)
    其中 c_j = ∑(p_(i,j)log_2(p_(i,j)))
    '''

    ## n是不同输入值的数量
    ## m是不同输出值的数量
    if n*m == 0:
        print('输入和输出值的范围必须大于零')
        return 'failed', np.nan, np.nan

    ## x是输入信号X(t)的概率分布
    x = cp.Variable(shape=n)

    ## y是输出信号Y(t)的概率分布
    ## P是信道过渡矩阵
    y = P@x

    ## I是x和y之间的互信息
    c = np.sum(np.array((xlogy(P, P) / math.log(2))), axis=0)
    I = c@x + cp.sum(cp.entr(y) / math.log(2))

    ## 通过最大化互信息进行最大化信道容量
    obj = cp.Maximize(I)
    constraints = [cp.sum(x) == sum_x, x >= 0]

    ## 构造并解决问题
    prob = cp.Problem(obj,constraints)
    prob.solve()
    if prob.status == 'optimal':
        return prob.status, prob.value, x.value
    else:
        return prob.status, np.nan, np.nan





np.set_printoptions(precision=3)
n = 2
m = 2
P = np.array([[0.85,0.15],
             [0.25,0.75]])
stat, C, x = channel_capacity(n, m, P)
print('问题状态: ',stat)
print('最优值 C = {:.4g}'.format(C))
print('最优变量 x = \n', x)


















