# https://wenku.csdn.net/answer/7u4oktb81d



import numpy as np

def omp(A, y, k):
    """
    OMP算法的Python实现

    参数：
    A: 测量矩阵，形状为(m, n)
    y: 观测向量，形状为(m, 1)
    k: 稀疏度，即信号的非零元素个数

    返回：
    x: 重构的稀疏信号，形状为(n, 1)
    """

    m, n = A.shape
    residual = y.copy()  # 初始化残差
    support = []  # 初始化支持集合

    for _ in range(k):
        # 计算投影系数
        projections = np.abs(A.T @ residual)
        # 选择最相关的原子
        index = np.argmax(projections)
        support.append(index)
        # 更新估计信号
        x = np.linalg.lstsq(A[:, support], y, rcond=None)[0]
        # 更新残差
        residual = y - A[:, support] @ x

    # 构造稀疏信号
    x_sparse = np.zeros((n, 1))
    x_sparse[support] = x

    return x_sparse
