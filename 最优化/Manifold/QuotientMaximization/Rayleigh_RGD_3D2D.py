#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ============================================================
# 0. 全局绘图设置
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 12

# ============================================================
# 1. 瑞利商最小化问题的黎曼梯度下降算法
# ============================================================
def riemannian_gd_rayleigh_real(Q, x0, eta0=1.0, rho=0.5, c=1e-4, epsilon=1e-10, Kmax=500, max_backtrack=100, eta_min=1e-16):
    """
    在单位球面上求解瑞利商最小化问题：
        min x^T Q x
        s.t. ||x||_2 = 1
    """
    Q = 0.5 * (Q + Q.T)
    x = x0.astype(float).copy()
    x = x / np.linalg.norm(x)
    x_hist = [x.copy()]
    f_hist = []
    grad_hist = []
    eta_hist = []
    bt_hist = []
    exitflag = 2
    message = '达到最大迭代次数后停止。'
    for k in range(Kmax):
        f = float(x.T @ Q @ x)
        g = Q @ x - f * x
        g_norm = np.linalg.norm(g)
        f_hist.append(f)
        grad_hist.append(g_norm)
        if g_norm <= epsilon:
            exitflag = 1
            message = '梯度范数达到停止阈值，算法收敛。'
            break
        eta = eta0
        success = False
        bt_used = 0
        for bt in range(1, max_backtrack + 1):
            x_new = x - eta * g
            x_new = x_new / np.linalg.norm(x_new)
            f_new = float(x_new.T @ Q @ x_new)
            if f_new <= f - c * eta * g_norm ** 2:
                success = True
                bt_used = bt
                break
            eta = rho * eta
            if eta <= eta_min:
                bt_used = bt
                break
        if not success:
            exitflag = 0
            message = 'Armijo 线搜索由于数值精度问题停止。'
            break
        x = x_new
        x_hist.append(x.copy())
        eta_hist.append(eta)
        bt_hist.append(bt_used)
    fval = float(x.T @ Q @ x)
    info = dict()
    info['x_hist'] = np.array(x_hist)
    info['f_hist'] = np.array(f_hist)
    info['grad_hist'] = np.array(grad_hist)
    info['eta_hist'] = np.array(eta_hist)
    info['bt_hist'] = np.array(bt_hist)
    info['iter'] = len(f_hist)
    info['exitflag'] = exitflag
    info['message'] = message
    return x, fval, info

# ============================================================
# 2. 轨迹绘制辅助函数
# ============================================================
def xyz_to_theta_phi(x_hist):
    """将三维单位球面坐标转换为二维展开图中的 theta 和 phi。"""
    x_hist = np.asarray(x_hist)
    x1 = x_hist[:, 0]
    x2 = x_hist[:, 1]
    x3 = x_hist[:, 2]
    theta_hist = np.arccos(np.clip(x3, -1.0, 1.0))
    phi_hist = np.arctan2(x2, x1)
    phi_hist = np.mod(phi_hist, 2 * np.pi)
    return theta_hist, phi_hist

def plot_trajectory_3d(ax, x_hist):
    """在三维单位球面图上绘制 RGD 轨迹。"""
    ax.plot(x_hist[:, 0], x_hist[:, 1], x_hist[:, 2], color='k', linewidth=2.4, marker='o', markersize=3, zorder=100000, label='RGD path')
    ax.scatter(x_hist[0, 0], x_hist[0, 1], x_hist[0, 2], color='lime', s=80, edgecolors='k', zorder=100001, label='Start')
    ax.scatter(x_hist[-1, 0], x_hist[-1, 1], x_hist[-1, 2], color='red', s=90, edgecolors='k', zorder=100002, label='End')

def plot_trajectory_2d(ax, phi_hist, theta_hist):
    """在二维 phi-theta 展开图上绘制 RGD 轨迹。"""
    phi_hist = np.asarray(phi_hist)
    theta_hist = np.asarray(theta_hist)
    if len(phi_hist) <= 1:
        ax.plot(phi_hist, theta_hist, color='k', linewidth=2.4, marker='o', markersize=3, zorder=100000, label='RGD path')
    else:
        start_idx = 0
        first_segment = True
        for i in range(1, len(phi_hist)):
            if abs(phi_hist[i] - phi_hist[i - 1]) > np.pi:
                if first_segment:
                    ax.plot(phi_hist[start_idx:i], theta_hist[start_idx:i], color='k', linewidth=2.4, marker='o', markersize=3, zorder=100000, label='RGD path')
                    first_segment = False
                else:
                    ax.plot(phi_hist[start_idx:i], theta_hist[start_idx:i], color='k', linewidth=2.4, marker='o', markersize=3, zorder=100000)
                start_idx = i
        if first_segment:
            ax.plot(phi_hist[start_idx:], theta_hist[start_idx:], color='k', linewidth=2.4, marker='o', markersize=3, zorder=100000, label='RGD path')
        else:
            ax.plot(phi_hist[start_idx:], theta_hist[start_idx:], color='k', linewidth=2.4, marker='o', markersize=3, zorder=100000)
    ax.scatter(phi_hist[0], theta_hist[0], color='lime', s=70, edgecolors='k', zorder=100001, label='Start')
    ax.scatter(phi_hist[-1], theta_hist[-1], color='red', s=80, edgecolors='k', zorder=100002, label='End')

# ============================================================
# 3. 生成单位球面网格并计算瑞利商
# ============================================================
intervals = 50
ntheta = intervals
nphi = 2 * intervals
theta = np.linspace(0, np.pi, ntheta + 1)
phi = np.linspace(0, 2 * np.pi, nphi + 1)
r = 1
pp_, tt_ = np.meshgrid(phi, theta)
Z = r * np.cos(tt_)
X = r * np.sin(tt_) * np.cos(pp_)
Y = r * np.sin(tt_) * np.sin(pp_)
Points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

Q = np.array([[1, 0.5, 1], [0.5, 2, -0.2], [1, -0.2, 1]], dtype=float)
Rayleigh_Q = np.sum((Points @ Q) * Points, axis=1)
Rayleigh_Q_ = np.reshape(Rayleigh_Q, X.shape)

# ============================================================
# 4. 运行黎曼梯度下降算法并记录迭代轨迹
# ============================================================
x0 = np.array([0.35, -0.88, 0.32], dtype=float)
x0 = x0 / np.linalg.norm(x0)
x_opt, f_opt, info = riemannian_gd_rayleigh_real(Q, x0, eta0=1.0, rho=0.5, c=1e-4, epsilon=1e-10, Kmax=500, max_backtrack=100, eta_min=1e-16)
x_hist = info['x_hist']
theta_hist, phi_hist = xyz_to_theta_phi(x_hist)

eigvals, eigvecs = np.linalg.eigh(Q)
lambda_min = eigvals[0]
v_min = eigvecs[:, 0]
direction_corr = abs(np.dot(x_opt, v_min))

print('算法状态:', info['message'])
print('迭代次数:', info['iter'])
print('估计得到的最小瑞利商:', f_opt)
print('真实最小特征值:', lambda_min)
print('最终点与最小特征向量的方向相关性:', direction_corr)

# x_opt 和 v_min 是实特征向量，方向只差一个符号，因此需要符号对齐
if np.dot(x_opt, v_min) < 0:
    x_aligned = -x_opt
else:
    x_aligned = x_opt

# 对齐后的误差与方向相关性
eigvec_err = np.linalg.norm(x_aligned - v_min)
direction_corr = abs(np.dot(x_opt, v_min) / (np.linalg.norm(x_opt) * np.linalg.norm(v_min)))

print('符号对齐后的特征向量误差:', eigvec_err)
print('最终点与最小特征向量的方向相关性:', direction_corr)
print('RGD 得到的最终点 x_opt:')
print(x_opt)
print('最小特征值对应的特征向量 v_min:')
print(v_min)
print('符号对齐后的 x_aligned:')
print(x_aligned)

# ============================================================
# 5. 原始三维图 1：球面网格 + 彩色散点 + 截面 + RGD 轨迹
# ============================================================
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='0.68', linewidth=0.25)
surf.set_facecolor((0, 0, 0, 0))
ax.scatter(X[::2, ::2], Y[::2, ::2], Z[::2, ::2], c=Rayleigh_Q_[::2, ::2], cmap='hsv', s=15)

# ## x轴切面
level_idx = 0.5
xx_, zz_ = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
ax.plot_surface(xx_, xx_ * 0 + level_idx, zz_, color='b', alpha=0.2)
ax.plot_wireframe(xx_, xx_ * 0 + level_idx, zz_, color='b', lw=0.2)
ax.contour(X, Y, Z, zdir='y', levels=[level_idx], linewidths=2, linestyles='--', colors='b')
plot_trajectory_3d(ax, x_hist)

ax.set_proj_type('ortho')
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# three reference lines
k_axis = 1.5
ax.plot((-k_axis, k_axis), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k_axis, k_axis), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k_axis, k_axis), 'k')
ax.axis('off')
ax.set_xlim((-np.max(np.abs(X)), np.max(np.abs(X))))
ax.set_ylim((-np.max(np.abs(Y)), np.max(np.abs(Y))))
ax.set_zlim((-np.max(np.abs(Z)), np.max(np.abs(Z))))
ax.set_box_aspect([1, 1, 1])
ax.view_init(azim=-155, elev=35)
ax.grid(False)
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 6. 原始三维图 2：球面颜色填充图 + RGD 轨迹
# ============================================================
norm_plt = plt.Normalize(Rayleigh_Q_.min(), Rayleigh_Q_.max())
colors = cm.RdYlBu_r(norm_plt(Rayleigh_Q_))

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors, linewidth=0.25, shade=False)
plot_trajectory_3d(ax, x_hist)

ax.set_proj_type('ortho')
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

k_axis = 1.5
ax.plot((-k_axis, k_axis), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k_axis, k_axis), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k_axis, k_axis), 'k')
ax.axis('off')
ax.set_xlim((-k_axis, k_axis))
ax.set_ylim((-k_axis, k_axis))
ax.set_zlim((-k_axis, k_axis))
ax.set_box_aspect([1, 1, 1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 7. 原始三维图 3：透明球面网格图 + RGD 轨迹
# ============================================================
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors, linewidth=0.25, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plot_trajectory_3d(ax, x_hist)

ax.set_proj_type('ortho')
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

k_axis = 1.5
ax.plot((-k_axis, k_axis), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k_axis, k_axis), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k_axis, k_axis), 'k')
ax.axis('off')
ax.set_xlim((-k_axis, k_axis))
ax.set_ylim((-k_axis, k_axis))
ax.set_zlim((-k_axis, k_axis))
ax.set_box_aspect([1, 1, 1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 8. 原始二维图 1：球面展开散点图 + RGD 轨迹
# ============================================================
fig, ax = plt.subplots()
c_scat = ax.scatter(pp_, tt_, c=Rayleigh_Q_, cmap='RdYlBu_r', s=1.8)
plot_trajectory_2d(ax, phi_hist, theta_hist)

ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 9. 原始二维图 2：伪彩色网格图 + RGD 轨迹
# ============================================================
fig, ax = plt.subplots()
c_mesh = ax.pcolormesh(pp_, tt_, Rayleigh_Q_, cmap='RdYlBu_r', shading='auto', vmin=Rayleigh_Q_.min(), vmax=Rayleigh_Q_.max())
fig.colorbar(c_mesh, ax=ax, shrink=0.58)
plot_trajectory_2d(ax, phi_hist, theta_hist)

ax.set_xlim(pp_.min(), pp_.max())
ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 10. 原始二维图 3：填充等高线图 + 白色等高线 + RGD 轨迹
# ============================================================
fig, ax = plt.subplots()
levels = np.linspace(Rayleigh_Q_.min(), Rayleigh_Q_.max(), 18)
colorbar = ax.contourf(pp_, tt_, Rayleigh_Q_, levels=levels, cmap='RdYlBu_r')
ax.contour(pp_, tt_, Rayleigh_Q_, levels=levels, colors='w')
fig.colorbar(colorbar, ax=ax, shrink=0.58)
plot_trajectory_2d(ax, phi_hist, theta_hist)

ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 11. 原始二维图 4：二维等高线图 + RGD 轨迹
# ============================================================
fig, ax = plt.subplots()
colorbar = ax.contour(pp_, tt_, Rayleigh_Q_, levels=levels, cmap='RdYlBu_r')
fig.colorbar(colorbar, ax=ax, shrink=0.58)
plot_trajectory_2d(ax, phi_hist, theta_hist)

ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 12. 从二维 theta-phi 平面中提取等高线
# ============================================================
fig_tmp, ax_tmp = plt.subplots()
all_contours = ax_tmp.contour(pp_, tt_, Rayleigh_Q_, levels=levels, cmap='RdYlBu_r')
plt.close(fig_tmp)

# ============================================================
# 13. 原始三维图 4：球面颜色图 + 白色等高线 + RGD 轨迹
# ============================================================
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
norm_plt = plt.Normalize(Rayleigh_Q_.min(), Rayleigh_Q_.max())
colors = cm.RdYlBu_r(norm_plt(Rayleigh_Q_))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors, linewidth=0.25, shade=False)

for level_idx, ctr_idx in zip(all_contours.levels, all_contours.allsegs):
    for i in range(len(ctr_idx)):
        phi_i = ctr_idx[i][:, 0]
        theta_i = ctr_idx[i][:, 1]
        Z_i = np.cos(theta_i)
        X_i = np.sin(theta_i) * np.cos(phi_i)
        Y_i = np.sin(theta_i) * np.sin(phi_i)
        ax.plot(X_i, Y_i, Z_i, color='w', linewidth=1, zorder=10000)

plot_trajectory_3d(ax, x_hist)

ax.set_proj_type('ortho')
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

k_axis = 1.5
ax.plot((-k_axis, k_axis), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k_axis, k_axis), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k_axis, k_axis), 'k')
ax.axis('off')
ax.set_xlim((-k_axis, k_axis))
ax.set_ylim((-k_axis, k_axis))
ax.set_zlim((-k_axis, k_axis))
ax.set_box_aspect([1, 1, 1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 14. 原始三维图 5：球面网格 + 彩色等高线 + RGD 轨迹
# ============================================================
norm = Normalize(vmin=all_contours.levels.min(), vmax=all_contours.levels.max(), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlBu_r)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color=[0.5, 0.5, 0.5], linewidth=0.5)
surf.set_facecolor((0, 0, 0, 0))

for level_idx, ctr_idx in zip(all_contours.levels, all_contours.allsegs):
    for i in range(len(ctr_idx)):
        phi_i = ctr_idx[i][:, 0]
        theta_i = ctr_idx[i][:, 1]
        Z_i = np.cos(theta_i)
        X_i = np.sin(theta_i) * np.cos(phi_i)
        Y_i = np.sin(theta_i) * np.sin(phi_i)
        ax.plot(X_i, Y_i, Z_i, color=mapper.to_rgba(level_idx), linewidth=3)

plot_trajectory_3d(ax, x_hist)

ax.set_proj_type('ortho')
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

k_axis = 1.5
ax.plot((-k_axis, k_axis), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k_axis, k_axis), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k_axis, k_axis), 'k')
ax.axis('off')
ax.set_xlim((-k_axis, k_axis))
ax.set_ylim((-k_axis, k_axis))
ax.set_zlim((-k_axis, k_axis))
ax.set_box_aspect([1, 1, 1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 15. 可选图 1：瑞利商随迭代次数的变化
# ============================================================
fig, ax = plt.subplots()
ax.plot(np.arange(1, len(info['f_hist']) + 1), info['f_hist'], linewidth=1.8, marker='o', markersize=3)
ax.grid(True)
ax.set_xlabel('Iteration')
ax.set_ylabel('Rayleigh quotient')
plt.show()

# ============================================================
# 16. 可选图 2：黎曼梯度范数随迭代次数的变化
# ============================================================
fig, ax = plt.subplots()
ax.semilogy(np.arange(1, len(info['grad_hist']) + 1), info['grad_hist'], linewidth=1.8, marker='o', markersize=3)
ax.grid(True)
ax.set_xlabel('Iteration')
ax.set_ylabel('Riemannian gradient norm')
plt.show()

# ============================================================
# 17. 可选图 3：Armijo 步长随迭代次数的变化
# ============================================================
if len(info['eta_hist']) > 0:
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, len(info['eta_hist']) + 1), info['eta_hist'], linewidth=1.8, marker='o', markersize=3)
    ax.grid(True)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Step size')
    plt.show()

# ============================================================
# 18. 可选图 4：每次迭代的 Armijo 回溯次数
# ============================================================
if len(info['bt_hist']) > 0:
    fig, ax = plt.subplots()
    ax.stem(np.arange(1, len(info['bt_hist']) + 1), info['bt_hist'])
    ax.grid(True)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Backtracking number')
    plt.show()
