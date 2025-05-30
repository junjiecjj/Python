



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

#%% Bk4_Ch12_01.py
x1 = np.arange(-2, 2, 0.05)
x2 = np.arange(-2, 2, 0.05)

xx1_fine, xx2_fine = np.meshgrid(x1, x2)
a = 1; b = 0; c = 2;
yy_fine = a*xx1_fine**2 + 2*b*xx1_fine*xx2_fine + c*xx2_fine**2

# 3D visualization
fig, ax = plt.subplots()
ax = plt.axes(projection='3d')

ax.plot_wireframe(xx1_fine, xx2_fine, yy_fine, color = [0.8,0.8,0.8], linewidth = 0.25)
ax.contour3D(xx1_fine,xx2_fine,yy_fine,15, cmap = 'RdYlBu_r')
ax.view_init(elev=30, azim=60)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
plt.tight_layout()
ax.set_proj_type('ortho')
plt.show()

# 2D visualization
fig, ax = plt.subplots()
ax.contourf(xx1_fine, xx2_fine, yy_fine, 15, cmap = 'RdYlBu_r')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_aspect('equal')

plt.show()

#%% Bk4_Ch12_02.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math as m

cos_theta_12 = np.cos(m.radians(135))
cos_theta_13 = np.cos(m.radians(60))
cos_theta_23 = np.cos(m.radians(120))

P = np.array([[1, cos_theta_12, cos_theta_13],
              [cos_theta_12, 1, cos_theta_23],
              [cos_theta_13, cos_theta_23, 1]])

L = np.linalg.cholesky(P)
R = L.T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.plot(0,0,0,color = 'r', marker = 'x', markersize = 12)
colors = ['b', 'r', 'g']
for i in np.arange(0,3):
    vector = R[:,i]
    v = np.array([vector[0],vector[1],vector[2]])
    vlength=np.linalg.norm(v)
    ax.quiver(0,0,0,vector[0],vector[1],vector[2], length=vlength, color = colors[i])

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.view_init(35, 60)
ax.set_proj_type('ortho')
ax.set_box_aspect([1,1,1])


#%% 协方差矩阵的特征值、相关性系数矩阵、余弦相似度矩阵、cholesky分解、
# SIGMA = D@P@D
import scipy
# Load the iris data
iris_sns = sns.load_dataset("iris")
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target
feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']
# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

# 余弦相似度矩阵 (cosine similarity matrix), C != rho
G = X.T@X
S = np.diag(np.linalg.norm(X, ord = 2, axis=0))
C = scipy.linalg.inv(S) @ G @ scipy.linalg.inv(S)

# 中心化
Xc = X - X.mean(axis = 0)
# 计算协方差矩阵
XXT = np.matrix(Xc.T) * np.matrix(Xc) / (len(Xc)-1)
#  Heatmap of covariance matrix
#  协方差矩阵
SIGMA = np.array(X_df.cov()) # == XXT
D = np.diag(np.sqrt(np.diag(SIGMA)))

#  相关性系数矩阵
RHO = np.array(X_df.corr()) # == rho
rho = np.corrcoef(X.T)

# 协方差矩阵的特征值分解
Lambda_, V_ = np.linalg.eig(SIGMA)
Lambda = np.diag(Lambda_)

# 相关性系数矩阵的特征值分解
Lambda1_, V1_ = np.linalg.eig(RHO)  # V1_ != V1
Lambda1 = np.diag(Lambda1_) ## != Lambda
# 相关性系数矩阵的特征值和协方差矩阵的特征值分解特征值不同，且特征向量也不同

# 鸢尾花四特征协方差矩阵热图
fig, axs = plt.subplots()
h = sns.heatmap(SIGMA, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f')
h.set_aspect("equal")
h.set_title('Covariance matrix')

#  鸢尾花数据相关性系数矩阵热图
fig, axs = plt.subplots()
h = sns.heatmap(RHO, cmap='rainbow', linewidths=.05,annot=True)
h.set_aspect("equal")
h.set_title('Correlation matrix')

# 余弦相似度矩阵热图
fig, axs = plt.subplots()
h = sns.heatmap(C, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f')
h.set_aspect("equal")
h.set_title('cosine similarity matrix')


#  协方差矩阵和相关性矩阵关系热图
print(f"Sigma - D@P@D = \n{np.array(SIGMA) - D@np.array(RHO)@D}")

fig, axs = plt.subplots(1, 7, figsize=(16, 4), constrained_layout = True)
plt.sca(axs[0])
ax = sns.heatmap(SIGMA, cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\Sigma$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(D, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('D')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(RHO, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('P')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(D, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('D')

# 图  协方差矩阵特征值分解
fig, axs = plt.subplots(1, 7, figsize=(16, 4), constrained_layout = True)
plt.sca(axs[0])
ax = sns.heatmap(SIGMA,cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\Sigma$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(V_, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$V$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Lambda, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\Lambda$')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(V_.T, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$V^T$')


# 图  相关性系数矩阵的特征值分解
# 相关性系数矩阵的特征值分解
# 相关性系数矩阵的特征值和协方差矩阵的特征值分解特征值不同，且特征向量也不同
fig, axs = plt.subplots(1, 7, figsize=(16, 4), constrained_layout = True)
plt.sca(axs[0])
ax = sns.heatmap(RHO,cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$P$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(V1_, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$V_Z$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Lambda1, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\Lambda_Z$')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(V1_.T, cmap='rainbow', linewidths=.05, annot=True,fmt='.2f', cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$V_Z^T$')


























































































































































































































































































