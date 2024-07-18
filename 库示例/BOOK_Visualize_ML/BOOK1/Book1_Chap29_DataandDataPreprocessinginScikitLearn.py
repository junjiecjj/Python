



# sklearn.datasets.load_boston() 波士顿房价数据集，包含 506 个样本，每个样本有 13 个特征，常用 于回归任务。
# sklearn.datasets.load_iris() 鸢尾花数据集，包含 150 个样本，每个样本有 4 个特征，常用于分 类任务。
# sklearn.datasets.load_diabetes() 糖尿病数据集，包含 442 个样本，每个样本有 10 个特征，常用于回 归任务。
# sklearn.datasets.load_digits() 手写数字数据集，包含 1797 个样本，每个样本是一个 8x8 像素的图 像，常用于分类任务。
# sklearn.datasets.load_linnerud() 体能训练数据集，包含 20 个样本，每个样本有 3 个特 征，常用于多重输出回归任务。
# sklearn.datasets.load_wine() 葡萄酒数据集，包含 178 个样本，每个样本有 13 个特征，常用于分 类任务。
# sklearn.datasets.load_breast_cancer() 乳腺癌数据集，包含 569 个样本，每个样本有 30 个特征，常用于分 类任务。
# sklearn.datasets.fetch_olivetti_faces() 奥利维蒂人脸数据集，包含 400 张 64x64 像素的人脸图像，常用于 人脸识别任务。
# sklearn.datasets.fetch_lfw_people() 人脸数据集，包含 13233 张人脸图像，常用于人脸识别和验证任 务。

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 导入Scikit-Learn中鸢尾花数据
from sklearn.datasets import load_iris, load_diabetes, load_digits
import numpy as np
import pandas as pd

# 导入鸢尾花数据
iris = load_iris()
# 鸢尾花数据前4个特征，NumPy数组
X = iris.data


print(iris.feature_names)
# 鸢尾花数据标签：0、1、2

y = iris.target
print(np.unique(y))

# 鸢尾花文字标签
print(iris.target_names)

# 创建数据帧
X_df = pd.DataFrame(X, columns = ['X1','X2','X3','X4'])
X_df.head()

round(X_df.describe(),2)






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Scikit-Learn生成样本数据集
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_moons
from sklearn.datasets import make_blobs, make_classification


n_samples = 500

# 产生环形数据集
circles = make_circles(n_samples=n_samples,  factor=0.1, noise=0.1)
# 参数 noise 为添加到数据中的高斯噪声的标准差。 参数 factor 为内外圆之间的比例因子。factor 取值在 0 到 1 之间，1.0 表示两个圆重叠，0.0 表示完全分离的两个圆。

# 产生月牙形状数据集
moons = make_moons(n_samples=n_samples,  noise=0.1)

# 生成聚类数据集，可以指定样本数、特征数、簇数等
blobs = make_blobs(n_samples=n_samples,  centers = 4, cluster_std = 1.5)
print(blobs[0].shape)
print(blobs[1].shape)


# 几何变换
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X = np.dot(blobs[0], transformation)
rotated = (X, blobs[1])

# 不同稀疏程度
varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5])

# 用于测试分类算法的样本数据集
classif = make_classification(n_samples = n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)

datasets = [circles, moons, blobs, rotated, varied, classif]


# 可视化，有标签
fig, axes = plt.subplots(2,3,figsize=(16,12))
axes = axes.flatten()
for dataset_idx, ax_idx in zip(datasets, axes):
    X, y = dataset_idx
    # 标准化
    X = StandardScaler().fit_transform(X)
    ax_idx.scatter(X[:, 0], X[:, 1], s = 18, c = y, cmap = 'hsv', edgecolors="k")

    ax_idx.set_xlim(-3, 3)
    ax_idx.set_ylim(-3, 3)
    ax_idx.set_xticks(())
    ax_idx.set_yticks(())
    ax_idx.set_aspect('equal', adjustable='box')



# 可视化，无标签
fig, axes = plt.subplots(2,3,figsize=(16,12))
axes = axes.flatten()
for dataset_idx, ax_idx in zip(datasets, axes):
    X, y = dataset_idx
    X = StandardScaler().fit_transform(X)

    ax_idx.scatter(X[:, 0], X[:, 1], s=18, edgecolors="k")
    ax_idx.set_xlim(-3, 3)
    ax_idx.set_ylim(-3, 3)
    ax_idx.set_xticks(())
    ax_idx.set_yticks(())
    ax_idx.set_aspect('equal', adjustable='box')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 标准化完成特征缩放
# 导入包
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# 使用load_iris()函数加载数据集
iris = load_iris()
X = iris.data    # 特征矩阵
y = iris.target  # 标签数组


# 原始数据散点图
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], s = 18, c = y)
# 质心位置
ax.axvline(x = X[:, 0].mean(), c = 'r')
ax.axhline(y = X[:, 1].mean(), c = 'r')
ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.set_xlabel('Sepal length, cm')
ax.set_ylabel('Sepal width, cm')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.set_xbound(lower = -3, upper = 8)
ax.set_ybound(lower = -3, upper = 8)



from sklearn.preprocessing import StandardScaler
# 标准化特征数据矩阵
scaler = StandardScaler()
# scaler.mean_
# scaler.var_
X_z_score = scaler.fit_transform(X)


# 标准化数据散点图
fig, ax = plt.subplots()
ax.scatter(X_z_score[:, 0], X_z_score[:, 1], s = 18, c = y)
# 质心位置
ax.axvline(x = X_z_score[:, 0].mean(), c = 'r')
ax.axhline(y = X_z_score[:, 1].mean(), c = 'r')
ax.set_xlabel('Sepal length, z-score')
ax.set_ylabel('Sepal width, z-score')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.set_xbound(lower = -3, upper = 8)
ax.set_ybound(lower = -3, upper = 8)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 处理缺失值

from sklearn.datasets import load_iris
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# 导入鸢尾花数据
X, y = load_iris(as_frame=True, return_X_y=True)

# 引入缺失值
X_NaN = X.copy()

mask = np.random.uniform(0,1,size = X_NaN.shape)
mask = (mask <= 0.4)
X_NaN[mask] = np.NaN

iris_df_NaN = X_NaN.copy()
iris_df_NaN['species'] = y
print(iris_df_NaN.isnull().sum() * 100 / len(iris_df_NaN))

# 可视化缺失值位置
is_NaN = iris_df_NaN.isna()
fig, ax = plt.subplots()
ax = sns.heatmap(is_NaN, cmap='gray_r', cbar=False)

# 用kNN插补
knni = KNNImputer(n_neighbors=5)
X_NaN_kNN = knni.fit_transform(X_NaN)

iris_df_kNN = pd.DataFrame(X_NaN_kNN, columns=X_NaN.columns, index=X_NaN.index)
iris_df_kNN['species'] = y

sns.pairplot(iris_df_kNN, hue='species',  palette = "bright")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 处理离群值

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

# 生成数据
n_samples = 500
outliers_fraction = 0.10
n_outliers = int(outliers_fraction * n_samples)
n_inliers  = n_samples - n_outliers
X_outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, 2)) #  (50, 2)

np.random.RandomState(0)
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features = 2)

datasets = [
    make_blobs(centers=[[0, 0], [0, 0]], cluster_std = 0.5, **blobs_params)[0], # (450, 2)
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std = [0.5, 0.5], **blobs_params)[0], # (450, 2)
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std = [1.5, 0.3], **blobs_params)[0], # (450, 2)
    4.0 * (make_moons(n_samples = n_samples, noise = 0.05, random_state = 0)[0]- np.array([0.5, 0.25]))] # (500, 2)

# 处理离群值
anomaly_algorithms = [
     EllipticEnvelope(contamination=outliers_fraction, random_state=42),
     OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1),
     IsolationForest(contamination=outliers_fraction, random_state=42)]

# 网格化数据，用来绘制等高线
xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
xy = np.c_[xx.ravel(), yy.ravel()]
colors = np.array(["#377eb8", "#ff7f00"])

# 可视化
fig = plt.figure(figsize=(12,16))
plot_idx = 1
for idx, X in enumerate(datasets):
    print(X.shape)
    X = np.concatenate([X, X_outliers], axis=0)
    for algorithm in anomaly_algorithms:
        algorithm.fit(X)
        y_pred = algorithm.fit(X).predict(X) # (550,)

        ax = fig.add_subplot(4, 3, plot_idx); plot_idx += 1
        Z = algorithm.predict(xy)
        Z = Z.reshape(xx.shape)
        # 绘制边界
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")
        # 绘制散点数据集
        ax.scatter(X[:, 0], X[:, 1], s = 10, color=colors[(y_pred + 1) // 2])
        ax.set_xlim(-7, 7); ax.set_ylim(-7, 7)
        # ax.set_xticks(()); ax.set_yticks(())
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 将鸢尾花数据集拆分为训练集和测试集

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# 自定义可视化函数
def visualize(df):
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
    sns.heatmap(df.iloc[:,0:-1], cmap='RdYlBu_r', yticklabels = False, cbar=False, ax = axs[0])
    sns.heatmap(df.iloc[:,[-1]], cmap='Set3', yticklabels = False, cbar=False, ax = axs[1])

# 导入鸢尾花数据
X,y = load_iris(return_X_y = True)

# 转化为Pandas DataFrame
columns = ['Sepal length, X1', 'Sepal width, X2', 'Petal length, X3', 'Petal width, X4', 'Species']
df_full = pd.DataFrame(np.c_[X,y], columns = columns)
visualize(df_full)

# 拆分鸢尾花数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# 训练集
df_train = pd.DataFrame(np.c_[X_train, y_train], columns = columns)
visualize(df_train)

# 测试集
df_test = pd.DataFrame(np.c_[X_test, y_test], columns = columns)
visualize(df_test)
































































































































































































































































































































































































































































































































































































































































































































































































