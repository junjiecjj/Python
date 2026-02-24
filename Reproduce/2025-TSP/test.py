import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

def solve_iceberg_shaping(N, alpha, L, K_s1, f_hat, use_psl=True):
    """
    求解冰山谱形优化问题

    参数:
    N: 原始信号长度
    alpha: 滚降因子
    L: 上采样因子
    K_s1: 延迟区域索引集合
    f_hat: 频域向量序列 f_hat_{k+1} for k in K_s1
    use_psl: True使用PSL目标，False使用ISL目标

    返回:
    g_opt: 最优的时域滤波器系数
    """

    # 计算相关参数
    N_alpha = int(alpha * N)
    N_non_rolloff = N - N_alpha
    N_zeros = N_non_rolloff // 2
    N_ones = N_non_rolloff // 2

    # 构建tilde_g_k函数
    def get_tilde_g_k(g, k):
        """计算tilde_g_k = g_n + (1 - g_n)e^{-j2πk/L}"""
        phase = -2j * np.pi * k / L
        return g + (1 - g) * np.exp(phase)

    # 目标函数
    def objective(g):
        obj_terms = []
        for k in K_s1:
            tilde_g_k = get_tilde_g_k(g, k)
            # 计算 |f_hat_{k+1}^H tilde_g_k|^2
            term = np.abs(np.vdot(f_hat[k], tilde_g_k))**2
            obj_terms.append(term)

        if use_psl:
            return np.max(obj_terms)  # PSL目标
        else:
            return np.sum(obj_terms)  # ISL目标

    # 约束条件
    constraints = []

    # 约束(45): 前N_zeros个元素为0
    def constraint_zeros(g):
        return g[:N_zeros]

    # 约束(46): 后N_ones个元素为1
    def constraint_ones(g):
        return g[-N_ones:]

    # 约束(47): 单调递增
    def constraint_monotonic(g):
        return np.diff(g)

    # 约束(48): 总能量约束
    def constraint_energy(g):
        return np.sum(g) - N/2

    # 定义约束字典
    constraints = [
        {'type': 'eq', 'fun': constraint_zeros},    # g_n = 0 for n=1,...,N_zeros
        {'type': 'eq', 'fun': constraint_ones, 'args': (np.ones(N_ones),)},  # g_n = 1 for tail
        {'type': 'ineq', 'fun': constraint_monotonic},  # g_{n+1} - g_n ≥ 0
        {'type': 'eq', 'fun': constraint_energy}    # sum(g_n) = N/2
    ]

    # 边界约束: 0 ≤ g_n ≤ 1
    bounds = [(0, 1) for _ in range(N)]

    # 初始猜测 (满足单调性的线性插值)
    g0 = np.zeros(N)
    g0[:N_zeros] = 0
    g0[-N_ones:] = 1
    # 中间部分线性插值
    if N_alpha > 0:
        rolloff_indices = range(N_zeros, N - N_ones)
        g0[rolloff_indices] = np.linspace(0, 1, len(rolloff_indices))

    # 求解优化问题
    result = minimize(objective, g0, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000, 'ftol': 1e-8, 'disp': True})

    if result.success:
        print("优化成功!")
        print(f"最优目标值: {result.fun}")
        return result.x
    else:
        print(f"优化失败: {result.message}")
        return None

def plot_iceberg_shape(g_opt, alpha, N):
    """绘制冰山谱形"""
    plt.figure(figsize=(10, 6))
    plt.plot(g_opt, 'b-', linewidth=2, label='最优冰山谱形')

    N_alpha = int(alpha * N)
    N_non_rolloff = N - N_alpha
    N_zeros = N_non_rolloff // 2

    # 标记关键区域
    plt.axvline(x=N_zeros, color='r', linestyle='--', alpha=0.7, label='滚降开始')
    plt.axvline(x=N - N_zeros, color='g', linestyle='--', alpha=0.7, label='滚降结束')

    plt.xlabel('频率索引 n')
    plt.ylabel('g_n')
    plt.title(f'冰山谱形 (α={alpha})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 参数设置
    N = 64      # 原始信号长度
    alpha = 0.3  # 滚降因子
    L = 4       # 上采样因子
    K_s1 = list(range(10, 30))  # 延迟区域索引

    # 生成测试用的f_hat向量（实际应用中应根据具体问题设置）
    f_hat = {}
    for k in K_s1:
        # 创建具有特定结构的测试向量
        f_hat[k] = np.exp(1j * 2 * np.pi * np.random.rand(N))

    # 求解优化问题（使用PSL目标）
    print("求解PSL优化问题...")
    g_optimal_psl = solve_iceberg_shaping(N, alpha, L, K_s1, f_hat, use_psl=True)

    if g_optimal_psl is not None:
        # 绘制结果
        plot_iceberg_shape(g_optimal_psl, alpha, N)

        # 验证约束满足情况
        print(f"前{N//5}个元素: {g_optimal_psl[:N//5]}")
        print(f"后{N//5}个元素: {g_optimal_psl[-N//5:]}")
        print(f"总能量: {np.sum(g_optimal_psl)} (目标: {N/2})")
        print(f"单调性检查: {np.all(np.diff(g_optimal_psl) >= -1e-6)}")
