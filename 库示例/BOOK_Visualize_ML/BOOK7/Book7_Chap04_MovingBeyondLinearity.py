


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 多项式回归

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# 生成随机数据
np.random.seed(0)
num = 30
X = np.random.uniform(0,4,num) # (30,)
y = np.sin(0.4*np.pi * X) + 0.4 * np.random.randn(num) # (30,)
data = np.column_stack([X,y]) # (30, 2)

x_array = np.linspace(0,4,101).reshape(-1,1)  # (101, 1)
degree_array = [1,2,3,4,7,8]

fig, axes = plt.subplots(3,2,figsize=(10,20))
axes = axes.flatten()
for ax, degree_idx in zip(axes,degree_array):
    poly = PolynomialFeatures(degree = degree_idx)
    X_poly = poly.fit_transform(X.reshape(-1, 1)) # (30, [2,3,4,5,8,9])

    # 训练线性回归模型
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    y_poly_pred = poly_reg.predict(X_poly)
    data_ = np.column_stack([X,y_poly_pred])
    y_array_pred = poly_reg.predict(poly.fit_transform(x_array))

    # 绘制散点图
    ax.scatter(X, y, s=20)
    ax.scatter(X, y_poly_pred, marker = 'x', color='k')
    ax.plot(([i for (i,j) in data_], [i for (i,j) in data]), ([j for (i,j) in data_], [j for (i,j) in data]), c=[0.6,0.6,0.6], alpha = 0.5)

    ax.plot(x_array, y_array_pred, color='r')
    ax.set_title('Degree = %d' % degree_idx)

    # 提取参数
    coef = poly_reg.coef_
    intercept = poly_reg.intercept_
    # 回归解析式
    equation = '$y = {:.1f}'.format(intercept)
    for j in range(1, len(coef)):
        equation += ' + {:.1f}x^{}'.format(coef[j], j)
    equation += '$'
    equation = equation.replace("+ -", "-")
    ax.text(0.05, -1.8, equation)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0,4)
    ax.grid(False)
    ax.set_ylim(-2,2)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 非线性回归


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer


p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

np.random.seed(123)

def func_exp(x, a, b, c):
    # exponential function
    return a * np.exp(b * x) + c

def func_log(x, a, b, c):
    # log function
    return a * np.log(b * x) + c

def generate_data(func, *args, noise=0):
    # generate data
    xs = np.linspace(1, 6, 50)
    ys = func(xs, *args)
    noise = noise * np.random.normal(size=len(xs)) + noise
    xs = xs.reshape(-1, 1)
    ys = (ys + noise).reshape(-1, 1)
    return xs, ys

#>>>>>>>>>>>>>>>>>>>>>>>>> 图 5. 对数-线性模型
# Generate data
x_samp, y_samp = generate_data(func_exp, 2.5, 1.2, 0.7, noise=10)
transformer = FunctionTransformer(np.log, validate=True)
y_trans = transformer.fit_transform(y_samp)

## Regression
regressor = LinearRegression()
results = regressor.fit(x_samp, y_trans)
y_fit = results.predict(x_samp)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(x_samp, y_samp)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(x_samp, y_samp)
plt.yscale('log')
plt.ylabel('ln(y)')
plt.xlabel('x')
plt.grid(True)
plt.tight_layout()

## fitted data
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(x_samp, y_samp, label = 'Data')
plt.plot(x_samp, np.exp(y_fit), "r", label="Fitted")
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(x_samp, y_samp, label = 'Data')
plt.plot(x_samp, np.exp(y_fit), "r", label="Fitted")
plt.yscale('log')
plt.ylabel('ln(y)')
plt.xlabel('x')
plt.legend()
plt.grid(True)
plt.tight_layout()

#>>>>>>>>>>>>>>>>>>>>>>>>> 图 6. 类似“对数”形状的样本数据 图 7. 线性-对数模型
x_samp, y_samp = generate_data(func_log, 2.5, 1.2, 0.7, noise=0.3)
x_trans = transformer.fit_transform(x_samp)

## Regression
regressor = LinearRegression()
results = regressor.fit(x_trans, y_samp)
y_fit = results.predict(x_trans)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(x_samp, y_samp)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(x_samp, y_samp)
plt.xscale('log')
plt.xlabel('ln(x)')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()

## fitted data
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(x_samp, y_samp, label = 'Data')
plt.plot(x_samp, y_fit, "r", label="Fitted")
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(x_samp, y_samp, label = 'Data')
plt.plot(x_samp, y_fit, "r", label="Fitted")
plt.xscale('log')
plt.xlabel('ln(x)')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 多项式回归

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
np.random.seed(0)


def true_fun(X):
    return np.cos(1.5 * np.pi * X)

n_samples = 50
degrees = [1, 2, 3, 4]
# degrees = [12, 13, 14, 15]

X = np.sort(np.random.rand(n_samples)) # (50,)
y = true_fun(X) + np.random.randn(n_samples) * 0.1 # (50,)

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())
    polynomial_features = PolynomialFeatures(degree=degrees[i],include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),("linear_regression", linear_regression)])

    pipeline.fit(X[:, np.newaxis], y) # (50, 1)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), color = 'r', label="Fitted")

    plt.scatter(X, y, edgecolor='b', s=20, label="Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")

    plt.title("Degree {}".format(degrees[i]))
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 逻辑函数
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

x = np.linspace(-5, 5, 100)
f_x = expit(x)


# Plot the logistic function

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(x, f_x, 'b-')
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_xlim(-5, 5)
plt.show()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 逻辑函数分类，一元

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


X_, y = load_iris(return_X_y=True)
X = X_[:,0]
X = X[:, np.newaxis] # (150, 1)
# y.shape
# Out[43]: (150,)

# KDE distributions of three classes

import seaborn as sns
iris = sns.load_dataset("iris")


fig, ax = plt.subplots()
sns.kdeplot(data=iris[['sepal_length','species']], x='sepal_length', hue = 'species', palette = "viridis",linewidth=1, fill=True)
plt.xlabel('Sepal length, $x_1$')
plt.xticks([4,5,6,7,8])


fig, ax = plt.subplots()
plt.scatter(X.ravel(), y, s = 18, alpha = 1, label = 'Original')
plt.ylabel('Real y')
plt.xlabel('Sepal length, $x_1$')
plt.yticks([0,1,2])
plt.xticks([4,5,6,7,8])
plt.legend()
plt.tight_layout()
plt.show()

#%% logistic regression
import numpy as np

clf = LogisticRegression()
clf.fit(X, y)

X_test = np.linspace(X.min()*0.9,X.max()*1.1,num = 100) # (100, )
X_test = X_test[:, np.newaxis] # (100, 1)

y_hat = clf.predict(X_test) # (100,)
y_prob = clf.predict_proba(X_test) # (100, 3)

b1 = clf.coef_
b0 = clf.intercept_


#%% probabilities
x = np.linspace(X.min()*0.9,X.max()*1.1,num = 100);

fig, ax = plt.subplots()
plt.plot(X_test, y_prob[:,0], color='r', linewidth=1, label = 'Class 0')
plt.fill_between(x, y_prob[:,0], color='r', alpha = 0.5)

plt.plot(X_test, y_prob[:,1], color='b', linewidth=1, label = 'Class 1')
plt.fill_between(x, y_prob[:,1], color='b', alpha = 0.5)

plt.plot(X_test, y_prob[:,2], color='g', linewidth=1, label = 'Class 2')
plt.fill_between(x, y_prob[:,2], color='g', alpha = 0.5)

plt.ylabel('Probability')
plt.xlabel('Sepal length, $x_1$')
plt.xticks([4,5,6,7,8])
plt.legend()
plt.tight_layout()
plt.show()


#%% Predicted y
fig, ax = plt.subplots()
plt.scatter(X.ravel(), clf.predict(X), s = 8, alpha = 0.5, label = 'Predicted')

plt.ylabel('Predicted y')
plt.xlabel('Sepal length, $x_1$')
plt.yticks([0,1,2])
plt.xticks([4,5,6,7,8])
plt.legend()
plt.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 逻辑函数分类，二元
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X_, y = load_iris(return_X_y=True)
X = X_[:,:2] #  (150, 2)
# y.shape
# Out[59]: (150,)


iris = sns.load_dataset("iris")
ax = sns.jointplot(data=iris[['sepal_length','sepal_width','species']], x = 'sepal_length', y = 'sepal_width', hue = 'species', palette = "viridis")
ax.ax_joint.set_xlabel('Sepal length, $x_1$')
ax.ax_joint.set_xticks([4,5,6,7,8])
ax.ax_joint.set_ylabel('Sepal width, $x_2$')
ax.ax_joint.set_yticks([2,3,4,5])


ax = sns.jointplot(data=iris[['sepal_length','sepal_width','species']], x = 'sepal_length', y = 'sepal_width', hue = 'species', palette = "viridis", kind="kde")
ax.ax_joint.set_xlabel('Sepal length, $x_1$')
ax.ax_joint.set_xticks([4,5,6,7,8])
ax.ax_joint.set_ylabel('Sepal width, $x_2$')
ax.ax_joint.set_yticks([2,3,4,5])

## Logistic regression
import numpy as np
clf = LogisticRegression()
clf.fit(X, y)

X1_test = np.linspace(X[:,0].min()*0.9, X[:,0].max()*1.1, num = 101)
X2_test = np.linspace(X[:,1].min()*0.9, X[:,1].max()*1.1, num = 101)
xx1, xx2 = np.meshgrid(X1_test, X2_test) # (101, 101)
X_test = np.c_[xx1.ravel(), xx2.ravel()] # (10201, 2)

y_hat = clf.predict(X_test)
y_hat = y_hat.reshape(xx1.shape) # (101, 101)

y_prob = clf.predict_proba(X_test) # (10201, 3)

b1 = clf.coef_
b0 = clf.intercept_



## probabilities
levels = np.linspace(0,1,15)
for i in np.arange(3):
    prob_class_i = y_prob[:,i].reshape(xx1.shape)
    fig, ax = plt.subplots()
    contour_h = ax.contourf(xx1,xx2, prob_class_i, levels = levels, cmap='RdYlBu_r')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Sepal length, $x_1$')
    ax.set_ylabel('Sepal width, $x_2$')
    cb = plt.colorbar(contour_h)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot a basic wireframe.
    ax.plot_wireframe(xx1,xx2, prob_class_i, rstride=4, cstride=4, color = [0.7,0.7,0.7], linewidth = 0.25)
    contour_h = ax.contour3D(xx1,xx2, prob_class_i, levels = levels, cmap='RdYlBu_r')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('Sepal length, $x_1$')
    ax.set_ylabel('Sepal width, $x_2$')
    ax.set_zlabel('Probability')
    ax.set_proj_type('ortho')
    plt.tight_layout()
    cb = plt.colorbar(contour_h)



## Decision boundary
from matplotlib.colors import ListedColormap
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.
cmap_light = ListedColormap(rgb)

cmap_bold = [[255, 51, 0], [0, 153, 255],[138,138,138]]
cmap_bold = np.array(cmap_bold)/255.


# visualization
fig, ax = plt.subplots()

# plot decision regions
plt.contourf(xx1, xx2, y_hat, cmap=cmap_light)

# plot decision boundaries
plt.contour(xx1, xx2, y_hat, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)

# Plot data points
sns.scatterplot(x=iris['sepal_length'], y=iris['sepal_width'], hue=iris['species'], palette=cmap_bold, alpha=1.0, linewidth = 1, edgecolor=[1,1,1])

ax.set_xlabel('Sepal length, $x_1$')
ax.set_xticks([4,5,6,7,8])
ax.set_ylabel('Sepal width, $x_2$')
ax.set_yticks([2,3,4])







































































































































































































































































































































































