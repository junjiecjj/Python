




#%%>>>>>>>>>>>>>>>>>>>>>>>>> 梯度下降法 (Gradient Descent): 解析
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mod = "easy"
if mod == 'easy':
    funkind = 'convex fun'
    ## 定义要最小化的函数（简单的二次函数）
    def f(w):
        return w[0] ** 2/4 + w[1] ** 2
    ## 定义函数关于 x 和 y 的偏导数
    def grad_f(w):
        return np.array([w[0]/2.0, 2*w[1]])
elif mod == 'hard':
    funkind = 'non-convex fun'
    ## 定义要最小化的函数（简单的二次函数）
    def f(w):
        return (1-w[1]**5 + w[0]**5)*np.exp(-w[0]**2 - w[1]**2)
    ## 定义函数关于 x 和 y 的偏导数
    def grad_f(w):
        gx = (5 * w[0]**4 * np.exp(-w[0]**2-w[1]**2) - 2*w[0] * (1 - w[1]**5 + w[0]**5) * np.exp(-w[0]**2-w[1]**2))
        gy = (-5 * w[1]**4 * np.exp(-w[0]**2-w[1]**2) - 2*w[1] * (1-w[1]**5 + w[0]**5) * np.exp(-w[0]**2-w[1]**2))
        return np.array([gx, gy])

# 定义梯度下降算法
def Gradient_descent(theta_init, lr, num_iterations, perturbation = 0.1):
    # 初始化参数
    theta = theta_init
    history = []
    history.append([theta[0], theta[1], f(theta)])
    # 执行梯度下降迭代
    for i in  range (num_iterations):
        # 计算梯度
        grad = np.sign(grad_f(theta)) + perturbation * np.random.randn(*theta.shape)
        # 更新参数
        theta = theta - lr * grad
        # 保存参数的历史记录
        history.append([theta[0], theta[1], f(theta)])

    return theta[0], theta[1], f(theta), np.array(history)

if mod == 'easy':
    # 定义用于绘制函数的网格
    x_range = np.arange(-10 ,10 , 0.1 )
    y_range = np.arange(-10 ,10 , 0.1 )
    X, Y = np.meshgrid(x_range, y_range)
    W_array = np.vstack([X.ravel(), Y.ravel()])
    Z = f(W_array).reshape(X.shape)

    # 执行梯度下降并绘制结果
    theta_init = np.array([8, 8])
elif mod == 'hard':
    # 定义用于绘制函数的网格
    x_range = np.arange(-6 , 6 , 0.1 )
    y_range = np.arange(-6 , 6 , 0.1 )
    X, Y = np.meshgrid(x_range, y_range)
    W_array = np.vstack([X.ravel(), Y.ravel()])
    Z = f(W_array).reshape(X.shape)
    # 执行梯度下降并绘制结果
    theta_init = np.array([0.01, 0.01])

learning_rate = 0.1
num_iterations = 100
pb = 10

x_opt, y_opt, f_opt, history = Gradient_descent(theta_init, learning_rate, num_iterations, perturbation = pb)

## 3D
fig, axs = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
axs.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
axs.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

axs.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5, zorder = 2)
axs.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
axs.set_proj_type('ortho')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 2)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 2)
axs.set_zlabel(r'$\mathrm{f}(\mathrm{\theta_1},\mathrm{\theta_2}$)', fontdict = font, labelpad = 6 )
axs.set_title(f"GD on {funkind}, pb = {pb}, 1-bit Quant", fontdict = font, )

axs.tick_params(axis='both', direction='in', width = 3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
axs.ticklabel_format(style='sci', scilimits=(-1,2), axis='z')
# ax1.view_init(azim=-100, elev=20)

# 设置坐标轴线宽
axs.xaxis.line.set_lw(1)
axs.yaxis.line.set_lw(1)
axs.zaxis.line.set_lw(1)
axs.grid(False)

# 显示图形
out_fig = plt.gcf()
optimizer = "GD"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_3D_{mod}fun_1bit.eps', bbox_inches='tight', pad_inches=0, )
plt.show()


## 2D courter
fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
# CS = ax.contourf(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)
CS = axs.contour(X, Y, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

axs.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
axs.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 0)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 0)
axs.set_title(f"GD on {funkind}, pb = {pb}, 1-bit Quant", fontdict = font,  )

axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

axs.set_xlim(X.min(), X.max())
axs.set_ylim(Y.min(), Y.max())
axs.grid(False)

# 显示图形
out_fig = plt.gcf()
optimizer = "GD"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_2D_{mod}fun_1bit.eps', bbox_inches='tight', pad_inches=0,)
plt.show()



