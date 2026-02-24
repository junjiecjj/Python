#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 22:24:45 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_psd_hermitian_method1(n, seed=None):
    """
    方法1：通过 A^H A 构造半正定 Hermitian 矩阵

    参数:
    n: 矩阵维度
    seed: 随机种子

    返回:
    H: n×n 半正定 Hermitian 矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成随机复数矩阵
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # 构造半正定 Hermitian 矩阵: A^H A
    H = A.conj().T @ A

    return H

def generate_psd_hermitian_method2(n, seed=None):
    """
    方法2：通过特征值分解构造

    参数:
    n: 矩阵维度
    seed: 随机种子

    返回:
    H: n×n 半正定 Hermitian 矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成随机酉矩阵
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)  # Q 是酉矩阵

    # 生成非负特征值
    eigenvalues = np.random.uniform(0.1, 10, n)

    # 构造矩阵: Q Λ Q^H
    H = Q @ np.diag(eigenvalues) @ Q.conj().T

    return H

def generate_psd_hermitian_method3(n, seed=None):
    """
    方法3：通过随机对角占优矩阵构造

    参数:
    n: 矩阵维度
    seed: 随机种子

    返回:
    H: n×n 半正定 Hermitian 矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成随机 Hermitian 矩阵
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = (A + A.conj().T) / 2  # 确保 Hermitian

    # 使其对角占优以确保半正定性
    for i in range(n):
        H[i, i] = np.sum(np.abs(H[i, :])) + np.random.uniform(0.1, 1)

    return H

def generate_well_conditioned_psd(n, max_condition_number=10, seed=None):
    """
    生成良条件的半正定 Hermitian 矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成随机酉矩阵
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)

    # 控制特征值范围以获得良条件数
    min_eig = 1.0
    max_eig = min_eig * max_condition_number
    eigenvalues = np.random.uniform(min_eig, max_eig, n)

    H = Q @ np.diag(eigenvalues) @ Q.conj().T
    return H

def generate_low_rank_psd(n, rank, seed=None):
    """
    生成低秩半正定 Hermitian 矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成 rank × n 随机矩阵
    A = np.random.randn(rank, n) + 1j * np.random.randn(rank, n)

    # 构造低秩半正定矩阵: A^H A
    H = A.conj().T @ A

    return H

def verify_hermitian_properties(matrix, tolerance=1e-10):
    """
    验证矩阵的 Hermitian 和半正定性

    参数:
    matrix: 待验证矩阵
    tolerance: 数值容差

    返回:
    is_hermitian: 是否是 Hermitian 矩阵
    is_psd: 是否是半正定矩阵
    eigenvalues: 特征值
    """
    # 验证 Hermitian 性质: H = H^H
    is_hermitian = np.allclose(matrix, matrix.conj().T, atol=tolerance)

    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(matrix)  # 对 Hermitian 矩阵使用 eigvalsh

    # 验证半正定性: 所有特征值 >= 0
    is_psd = np.all(eigenvalues >= -tolerance)

    return is_hermitian, is_psd, eigenvalues

def analyze_matrix_properties(H, name="Matrix"):
    """
    分析矩阵性质并打印结果
    """
    print(f"\n{name} 性质分析:")
    print("=" * 40)

    is_hermitian, is_psd, eigenvalues = verify_hermitian_properties(H)

    print(f"维度: {H.shape}")
    print(f"是 Hermitian 矩阵: {is_hermitian}")
    print(f"是半正定矩阵: {is_psd}")
    print(f"特征值范围: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
    print(f"条件数: {np.linalg.cond(H):.6f}")
    print(f"迹(Trace): {np.trace(H):.6f}")
    print(f"Frobenius 范数: {np.linalg.norm(H, 'fro'):.6f}")

    return eigenvalues


def plot_matrix_properties(H, eigenvalues, title="Hermitian 半正定矩阵"):
    """
    可视化矩阵和其特征值
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 绘制矩阵实部
    im1 = axes[0, 0].imshow(H.real, cmap='RdBu_r', aspect='equal')
    axes[0, 0].set_title(f'{title} - 实部')
    plt.colorbar(im1, ax=axes[0, 0])

    # 绘制矩阵虚部
    im2 = axes[0, 1].imshow(H.imag, cmap='RdBu_r', aspect='equal')
    axes[0, 1].set_title(f'{title} - 虚部')
    plt.colorbar(im2, ax=axes[0, 1])

    # 绘制特征值分布
    axes[1, 0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('特征值分布')
    axes[1, 0].set_xlabel('特征值索引')
    axes[1, 0].set_ylabel('特征值')
    axes[1, 0].grid(True, alpha=0.3)

    # 绘制特征值直方图
    axes[1, 1].hist(eigenvalues, bins=min(20, len(eigenvalues)), alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('特征值直方图')
    axes[1, 1].set_xlabel('特征值')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compare_generation_methods(n=5):
    """
    比较不同生成方法
    """
    print("半正定 Hermitian 矩阵生成方法比较")
    print("=" * 50)

    methods = [
        ("方法1: A^H A", generate_psd_hermitian_method1),
        ("方法2: 特征值分解", generate_psd_hermitian_method2),
        ("方法3: 对角占优", generate_psd_hermitian_method3)
    ]

    matrices = []
    all_eigenvalues = []

    for method_name, method_func in methods:
        H = method_func(n, seed=42)
        matrices.append(H)

        print(f"\n{method_name}:")
        eigenvalues = analyze_matrix_properties(H, method_name)
        all_eigenvalues.append(eigenvalues)

        # 显示矩阵的一部分
        print("矩阵 (前3×3):")
        print(H[:3, :3])

    return matrices, all_eigenvalues, methods


def comprehensive_example():
    """
    综合示例：生成和分析半正定 Hermitian 矩阵
    """
    # 设置参数
    n = 6  # 矩阵维度
    seed = 42

    print("半正定 Hermitian 矩阵生成示例")
    print("=" * 50)

    # 比较不同方法
    matrices, eigenvalues_list, methods = compare_generation_methods(n)

    # 可视化每个方法的结果
    for i, (method_name, H) in enumerate(zip([m[0] for m in methods], matrices)):
        plot_matrix_properties(H, eigenvalues_list[i], method_name)

    # 特殊应用：生成特定条件的矩阵
    print("\n特殊应用示例")
    print("=" * 30)

    # 生成条件数较小的矩阵
    H_well_conditioned = generate_well_conditioned_psd(n, max_condition_number=10, seed=seed)
    analyze_matrix_properties(H_well_conditioned, "良条件矩阵")

    # 生成低秩矩阵
    H_low_rank = generate_low_rank_psd(n, rank=2, seed=seed)
    analyze_matrix_properties(H_low_rank, "低秩矩阵")


# 运行示例
if __name__ == "__main__":
    comprehensive_example()

    # 额外测试：大规模矩阵
    print("\n大规模矩阵测试 (20×20)")
    print("=" * 40)
    H_large = generate_psd_hermitian_method1(20, seed=42)
    analyze_matrix_properties(H_large, "20×20 大规模矩阵")

    # 验证数学性质
    print("\n数学性质验证")
    print("=" * 30)
    H_test = generate_psd_hermitian_method2(4, seed=42)

    # 验证二次型非负
    x = np.random.randn(4) + 1j * np.random.randn(4)
    quadratic_form = x.conj().T @ H_test @ x
    print(f"随机向量 x 的二次型值: {quadratic_form:.6f}")
    print(f"二次型是实数: {np.isclose(quadratic_form.imag, 0)}")
    print(f"二次型非负: {quadratic_form.real >= -1e-10}")
