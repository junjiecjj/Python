import numpy as np
import matplotlib.pyplot as plt

def generate_sigma_s(N=10, K=10, delta_k=1.0):
    """
    生成感知信道方差矩阵 Σ_s

    参数:
    N: int, 发射天线数量
    K: int, 路径数量
    delta_k: float, 路径损耗期望值

    返回:
    Sigma_s: numpy array, 形状为 (N, N) 的协方差矩阵
    theta_k: numpy array, 选择的K个角度（度）
    """
    # 随机选择K个角度在[-90°, 90°]区间内
    theta_k_deg = np.random.uniform(-90, 90, K)
    theta_k_rad = np.deg2rad(theta_k_deg)

    # 初始化 Sigma_s
    Sigma_s = np.zeros((N, N), dtype=complex)

    # 对每个路径计算贡献
    for k in range(K):
        # 生成阵列导向矢量 a(θ_k)
        a_theta =  (1/np.sqrt(N)) * np.exp(1j * np.pi * np.arange(N) * np.sin(theta_k_rad[k]))

        # 计算外积并加到 Sigma_s
        Sigma_s += delta_k**2 * np.outer(a_theta, a_theta.conj())

    return Sigma_s, theta_k_deg


def verify_sigma_s_properties(Sigma_s):
    """
    验证生成的 Sigma_s 的性质
    """
    print("Sigma_s 性质验证:")
    print(f"矩阵形状: {Sigma_s.shape}")
    print(f"是否为 Hermitian (共轭对称): {np.allclose(Sigma_s, Sigma_s.conj().T)}")
    print(f"是否为半正定: {np.all(np.linalg.eigvals(Sigma_s) >= -1e-10)}")
    print(f"矩阵的秩: {np.linalg.matrix_rank(Sigma_s)}")
    print(f"特征值: {np.linalg.eigvals(Sigma_s)}")

def plot_steering_pattern(theta_k_deg, N):
    """
    绘制阵列导向矢量的方向图
    """
    theta_test = np.deg2rad(np.linspace(-90, 90, 181))
    patterns = []

    # for theta in theta_test:
        # a = generate_steering_vector(theta, N)
        # patterns.append(np.abs(a[0]))  # 取第一个元素的幅度

    plt.figure(figsize=(10, 6))
    plt.plot(np.rad2deg(theta_test), patterns, label=f'N={N}')
    plt.scatter(theta_k_deg, np.ones_like(theta_k_deg)*0.5, color='red',
                label=f'Selected angles (K={len(theta_k_deg)})', zorder=5)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Array Response')
    plt.title('Steering Vector Pattern')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 生成并测试 Sigma_s
if __name__ == "__main__":
    # 使用默认参数生成 Sigma_s
    Sigma_s, theta_k = generate_sigma_s(N=10, K=10, delta_k=1.0)

    print("生成的参数:")
    print(f"天线数量 N: 10")
    print(f"路径数量 K: 10")
    print(f"选择的角度 (度): {theta_k}")
    print(f"路径损耗 delta_k: 1.0")
    print("\n生成的 Sigma_s 矩阵:")
    print(f"形状: {Sigma_s.shape}")
    print(f"实部范围: [{Sigma_s.real.min():.3f}, {Sigma_s.real.max():.3f}]")
    print(f"虚部范围: [{Sigma_s.imag.min():.3f}, {Sigma_s.imag.max():.3f}]")

    # 验证性质
    verify_sigma_s_properties(Sigma_s)

    # 绘制方向图
    # plot_steering_pattern(theta_k, 10)

    # 显示矩阵结构
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 3, 1)
    # plt.imshow(Sigma_s.real, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Sigma_s - Real Part')

    # plt.subplot(1, 3, 2)
    # plt.imshow(Sigma_s.imag, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Sigma_s - Imaginary Part')

    # plt.subplot(1, 3, 3)
    # plt.imshow(np.abs(Sigma_s), cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Sigma_s - Magnitude')

    # plt.tight_layout()
    # plt.show()
