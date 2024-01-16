





#=================================================================
#        高斯信道中的最优功率和带宽分配
# https://www.wuzao.com/document/cvxpy/examples/applications/optimal_power_gaussian_channel_BV4.62.html
#=================================================================

import numpy as np
import cvxpy as cp



def optimal_power(n, a_val, b_val, P_tot=1.0, W_tot=1.0):
    # 输入参数: α 和 β 是 R_i 方程中的常数
    n = len(a_val)
    if n != len(b_val):
        print('alpha 和 beta 向量必须具有相同的长度！')
        return 'failed', np.nan, np.nan, np.nan

    P = cp.Variable(shape=n)
    W = cp.Variable(shape=n)
    alpha = cp.Parameter(shape=n)
    beta = cp.Parameter(shape=n)
    alpha.value = np.array(a_val)
    beta.value = np.array(b_val)

    # 这个函数将被用作目标函数，因此必须是 DCP 的；即，内部的元素乘法必须发生在 kl_div 内部， 而不是外部，否则求解器无法知道它是 DCP 的...
    ## 尽管这是一个凸优化问题，但由于 和是变量，必须将其重新写成 DCP 形式，因为 DCP 禁止直接将一个变量除以另一个变量。为了将问题重写为 DCP 格式，我们利用 CVXPY 中的函数来计算 Kullback-Leibler 散度。
    R = cp.kl_div(cp.multiply(alpha, W), cp.multiply(alpha, W + cp.multiply(beta, P))) - cp.multiply(alpha, cp.multiply(beta, P))

    objective = cp.Minimize(cp.sum(R))
    constraints = [ P >= 0.0,
                    W >= 0.0,
                    cp.sum(P) - P_tot == 0.0,
                    cp.sum(W) - W_tot == 0.0]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.status, -prob.value, P.value, W.value




np.set_printoptions(precision = 5)
n = 5               # 系统中接收器的数量

a_val = np.arange(10, n+10)/(1.0*n)  # α
b_val = np.arange(10, n+10)/(1.0*n)  # β
P_tot = 0.5
W_tot = 1.0
status, utility, power, bandwidth = optimal_power(n, a_val, b_val, P_tot, W_tot)

print('状态: {}'.format(status))
print('最优效用值 = {:.4g}'.format(utility))
print('最优功率水平:\n{}'.format(power))
print('最优带宽:\n{}'.format(bandwidth))







































































































































































