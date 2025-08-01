


#%% Bk5_Ch18_01

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

plt.close('all')

iris = load_iris()
# A copy from Sklearn

X_1_to_4 = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

X_df = pd.DataFrame(X_1_to_4, columns = feature_names)
y_df = pd.DataFrame(y, columns=['label'])
y_df = y_df.replace({0: 'C_1', 1: 'C_2', 2: 'C_3'})

X1_df = X_df['Sepal length, $X_1$']

#%% likelihood PDF, given class Y
#### 图 2. 三个似然概率 PDF 曲线 fX|Y(x|Ck)
# given C1 (y = 0)
x1 = np.linspace(4,8,161)
fig, ax = plt.subplots()
KDE_C1 = sm.nonparametric.KDEUnivariate(X1_df[y==0])
KDE_C1.fit(bw=0.1)
f_x1_given_C1 = KDE_C1.evaluate(x1)
print(f"f_x1_given_C1.sum() * (x1[1] - x1[0]) = {f_x1_given_C1.sum() * (x1[1] - x1[0])}") # 注意，fX|Y(x|Ck) 和横轴包裹的面积为 1

ax.fill_between(x1, f_x1_given_C1, facecolor = '#FF3300',alpha = 0.2)
ax.plot(x1, f_x1_given_C1,color = '#FF3300', label = '$f_{X1|Y}(x_1|C_1)$, likelihood')
ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])
ax.set_ylabel('PDF')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()

# given C2 (y = 1)
fig, ax = plt.subplots()
KDE_C2 = sm.nonparametric.KDEUnivariate(X1_df[y==1])
KDE_C2.fit(bw=0.1)
f_x1_given_C2 = KDE_C2.evaluate(x1)
print(f"f_x1_given_C2.sum() * (x1[1] - x1[0]) = {f_x1_given_C2.sum() * (x1[1] - x1[0])}") # 注意，fX|Y(x|Ck) 和横轴包裹的面积为 1

ax.fill_between(x1, f_x1_given_C2, facecolor = '#0099FF',alpha = 0.2)
ax.plot(x1, f_x1_given_C2,color = '#0099FF', label = '$f_{X1|Y}(x_1|C_2)$, likelihood')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])
ax.set_ylabel('PDF')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()

# given C3 (y = 2)
fig, ax = plt.subplots()

KDE_C3 = sm.nonparametric.KDEUnivariate(X1_df[y==2])
KDE_C3.fit(bw=0.1)
f_x1_given_C3 = KDE_C3.evaluate(x1)
print(f"f_x1_given_C3.sum() * (x1[1] - x1[0]) = {f_x1_given_C3.sum() * (x1[1] - x1[0])}")

ax.fill_between(x1, f_x1_given_C3, facecolor = '#8A8A8A',alpha = 0.2)
ax.plot(x1, f_x1_given_C3,color = '#8A8A8A', label = '$f_{X1|Y}(x_1|C_3)$, likelihood')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])
ax.set_ylabel('PDF')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()

## compare three likelihood curves
fig, ax = plt.subplots()

ax.fill_between(x1, f_x1_given_C1, facecolor = '#FF3300',alpha = 0.2)
ax.plot(x1, f_x1_given_C1,color = '#FF3300', label = '$f_{X1|Y}(x_1|C_1)$')

ax.fill_between(x1, f_x1_given_C2, facecolor = '#0099FF',alpha = 0.2)
ax.plot(x1, f_x1_given_C2,color = '#0099FF', label = '$f_{X1|Y}(x_1|C_2)$')

ax.fill_between(x1, f_x1_given_C3, facecolor = '#8A8A8A',alpha = 0.2)
ax.plot(x1, f_x1_given_C3,color = '#8A8A8A', label = '$f_{X1|Y}(x_1|C_3)$')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])
ax.set_ylabel('Likelihood PDF')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()

# 图 3. 150 个样本数据总三类的频数和先验概率
#%% prior probability
y_counts = y_df.value_counts()

#Plot the data:
my_colors = ['#FF3300', '#0099FF', '#8A8A8A']

fig, ax = plt.subplots()
y_counts.plot.bar(color=my_colors)
plt.ylabel('Count')
plt.show()

fig, ax = plt.subplots()
y_prob = y_counts/y_df.count().values[0]
y_prob.plot.bar(color=my_colors)
plt.ylabel('Prior probability')
plt.show()

#%% Joint PDF, 图 4. 先验概率和联合概率的关系

f_x1_joint_C1 = f_x1_given_C1*y_prob['C_1']
f_x1_joint_C2 = f_x1_given_C2*y_prob['C_2']
f_x1_joint_C3 = f_x1_given_C3*y_prob['C_3']

# C1
fig, ax = plt.subplots()

# Conditional likelihood
ax.plot(x1, f_x1_given_C1,color = '#FF3300', linestyle = '--', label = '$f_{X1|Y}(x_1|C_1)$, likelihood')

# Joint
ax.fill_between(x1, f_x1_joint_C1, facecolor = '#FF3300',alpha = 0.2)
ax.plot(x1, f_x1_joint_C1,color = '#FF3300', label = '$f_{X1,Y}(x_1,C_1)$, joint')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])
ax.set_ylabel('$PDF$')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()

# C2
fig, ax = plt.subplots()

# Conditional likelihood
ax.plot(x1, f_x1_given_C2,color = '#0099FF', linestyle = '--', label = '$f_{X1|Y}(x_1|C_2)$, likelihood')

# Joint
ax.fill_between(x1, f_x1_joint_C2, facecolor = '#0099FF',alpha = 0.2)
ax.plot(x1, f_x1_joint_C2,color = '#0099FF', label = '$f_{X1,Y}(x_1,C_2)$, joint')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])
ax.set_ylabel('$PDF$')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()

# C3
fig, ax = plt.subplots()

# Conditional likelihood
ax.plot(x1, f_x1_given_C3,color = '#8A8A8A', linestyle = '--', label = '$f_{X1|Y}(x_1|C_3)$, likelihood')

# Joint
ax.fill_between(x1, f_x1_joint_C3, facecolor = '#8A8A8A',alpha = 0.2)
ax.plot(x1, f_x1_joint_C3,color = '#8A8A8A', label = '$f_{X1,Y}(x_1,C_3)$, joint')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])
ax.set_ylabel('$PDF$')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()

# compare joint
fig, ax = plt.subplots()

# Joint
ax.fill_between(x1, f_x1_joint_C1, facecolor = '#FF3300',alpha = 0.2)
ax.plot(x1, f_x1_joint_C1,color = '#FF3300', label = '$f_{X1,Y}(x_1,C_1)$')

ax.fill_between(x1, f_x1_joint_C2, facecolor = '#0099FF',alpha = 0.2)
ax.plot(x1, f_x1_joint_C2,color = '#0099FF', label = '$f_{X1,Y}(x_1,C_2)$')

ax.fill_between(x1, f_x1_joint_C3, facecolor = '#8A8A8A',alpha = 0.2)
ax.plot(x1, f_x1_joint_C3,color = '#8A8A8A', label = '$f_{X1,Y}(x_1,C_3)$')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])
ax.set_ylabel('Conditional PDF')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()


#%% Evidence fX_1(x_1)
# 图 5. 叠加联合概率曲线，估算证据因子概率密度函数
f_x1 = f_x1_joint_C1 + f_x1_joint_C2 + f_x1_joint_C3
fig, ax = plt.subplots()

ax.plot(x1, f_x1, color = '#00448A', label = '$f_{X1}(x_1)$, evidence (marginal)')
ax.fill_between(x1, f_x1, facecolor = '#00448A',alpha = 0.1)
ax.plot(x1, f_x1_joint_C1,color = '#FF3300', label = '$f_{X1,Y}(x_1,C_1)$')
ax.plot(x1, f_x1_joint_C2,color = '#0099FF', label = '$f_{X1,Y}(x_1,C_2)$')
ax.plot(x1, f_x1_joint_C3,color = '#8A8A8A', label = '$f_{X1,Y}(x_1,C_3)$')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,0.75])
ax.set_yticks([0, 0.5])
ax.set_xlim([4,8])
ax.set_ylabel('PDF')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()

#%% Posterior. 图 7. 比较联合概率、证据因子、后验概率曲线
f_C1_given_x1 = f_x1_joint_C1/f_x1
f_C2_given_x1 = f_x1_joint_C2/f_x1
f_C3_given_x1 = f_x1_joint_C3/f_x1

# C1
fig, (ax1, ax2, ax3) = plt.subplots(3)

# joint C1
ax1.plot(x1, f_x1_joint_C1,color = '#FF3300', label = '$f_{x1, Y}(x_1, C_1)$, joint')
ax1.fill_between(x1, f_x1_joint_C1, facecolor = '#FF3300',alpha = 0.2)
ax1.set_ylim([0,1])
ax1.set_yticks([0, 0.5, 1])
ax1.set_xlim([4,8])
ax1.set_xticks([])
ax1.legend()

# marginal, x1
ax2.plot(x1, f_x1, color = '#00448A', label = '$f_{X1}(x_1)$, marginal')
ax2.fill_between(x1, f_x1, facecolor = '#00448A',alpha = 0.1)
ax2.set_ylim([0,1])
ax2.set_yticks([0, 0.5, 1])
ax2.set_xlim([4,8])
ax2.set_xticks([])
ax2.legend()

# given x1, probability of C1
ax3.fill_between(x1, f_C1_given_x1, facecolor = '#FF3300',alpha = 0.2)
ax3.plot(x1, f_C1_given_x1,color = '#FF3300', label = '$f_{Y|X1}(C_1|x_1)$, posterior')

ax3.autoscale(enable=True, axis='x', tight=True)
ax3.autoscale(enable=True, axis='y', tight=True)
ax3.set_ylim([0,1])
ax3.set_yticks([0, 0.5, 1])
ax3.set_xlim([4,8])
ax3.set_xlabel('Sepal length, $x_1$')
ax3.legend()
plt.show()

# C2
fig, (ax1, ax2, ax3) = plt.subplots(3)

# joint C2
ax1.plot(x1, f_x1_joint_C2,color = '#0099FF', label = '$f_{x1, Y}(x_1, C_2)$, joint')
ax1.fill_between(x1, f_x1_joint_C2, facecolor = '#0099FF',alpha = 0.2)
ax1.set_ylim([0,1])
ax1.set_yticks([0, 0.5, 1])
ax1.set_xlim([4,8])
ax1.set_xticks([])
ax1.legend()

# marginal, x1
ax2.plot(x1, f_x1, color = '#00448A', label = '$f_{X1}(x_1)$, marginal')
ax2.fill_between(x1, f_x1, facecolor = '#00448A',alpha = 0.1)
ax2.set_ylim([0,1])
ax2.set_yticks([0, 0.5, 1])
ax2.set_xlim([4,8])
ax2.set_xticks([])
ax2.legend()

# given x1, probability of C2
ax3.fill_between(x1, f_C2_given_x1, facecolor = '#0099FF',alpha = 0.2)
ax3.plot(x1, f_C2_given_x1,color = '#0099FF', label = '$f_{Y|X1}(C_2|x_1)$, posterior')

ax3.autoscale(enable=True, axis='x', tight=True)
ax3.autoscale(enable=True, axis='y', tight=True)
ax3.set_ylim([0,1])
ax3.set_yticks([0, 0.5, 1])
ax3.set_xlim([4,8])
ax3.set_xlabel('Sepal length, $x_1$')
ax3.legend()
plt.show()

# C3
fig, (ax1, ax2, ax3) = plt.subplots(3)

# joint C3
ax1.plot(x1, f_x1_joint_C3,color = '#8A8A8A', label = '$f_{x1, Y}(x_1, C_3)$, joint')
ax1.fill_between(x1, f_x1_joint_C3, facecolor = '#8A8A8A',alpha = 0.2)
ax1.set_ylim([0,1])
ax1.set_yticks([0, 0.5, 1])
ax1.set_xlim([4,8])
ax1.set_xticks([])
ax1.legend()

# marginal, x1
ax2.plot(x1, f_x1, color = '#00448A', label = '$f_{X1}(x_1)$, marginal')
ax2.fill_between(x1, f_x1, facecolor = '#00448A',alpha = 0.1)
ax2.set_ylim([0,1])
ax2.set_yticks([0, 0.5, 1])
ax2.set_xlim([4,8])
ax2.set_xticks([])
ax2.legend()

# given x1, probability of C3
ax3.fill_between(x1, f_C3_given_x1, facecolor = '#8A8A8A',alpha = 0.2)
ax3.plot(x1, f_C3_given_x1,color = '#8A8A8A', label = '$f_{Y|X1}(C_3|x_1)$, posterior')

ax3.autoscale(enable=True, axis='x', tight=True)
ax3.autoscale(enable=True, axis='y', tight=True)
ax3.set_ylim([0,1])
ax3.set_yticks([0, 0.5, 1])
ax3.set_xlim([4,8])
ax3.set_xlabel('Sepal length, $x_1$')
ax3.legend()
plt.show()


#%% 图 8. 比较三个后验概率 PDF 曲线 fY|X(C1|x)、fY|X(C2|x)、fY|X(C3|x)
fig, ax = plt.subplots()

ax.fill_between(x1, f_C1_given_x1, facecolor = '#FF3300',alpha = 0.2)
ax.plot(x1, f_C1_given_x1,color = '#FF3300', label = '$f_{Y|X1}(C_1|x_1)$')

ax.fill_between(x1, f_C2_given_x1, facecolor = '#0099FF',alpha = 0.2)
ax.plot(x1, f_C2_given_x1,color = '#0099FF', label = '$f_{Y|X1}(C_2|x_1)$')

ax.fill_between(x1, f_C3_given_x1, facecolor = '#8A8A8A',alpha = 0.2)
ax.plot(x1, f_C3_given_x1,color = '#8A8A8A', label = '$f_{Y|X1}(C_3|x_1)$')

ax.axhline(y = 1, color = 'k', linestyle = '--')

ax.autoscale(enable=True, axis='x', tight=True)
ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim([0,1])
ax.set_yticks([0, 0.5, 1])
ax.set_xlim([4,8])
ax.set_ylabel('Posterior probability')
ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()
plt.close()


#%% compare posterior, likelihood, marginal (evidence), and joint
# 图 9. 比较后验概率 fY|X(C1 | x)、似然概率 fX|Y(x|C1)、证据因子 fX(x)、联合概率 fX,Y(x,C1)
## C1
fig, ax = plt.subplots()

# posterior
ax.plot(x1, f_C1_given_x1,color = 'r',label = '$f_{Y|X1}(C_1|x_1)$, posterior')

# likelihood
ax.plot(x1, f_x1_given_C1,color = '#0099FF',label = '$f_{X1|Y}(x_1|C_1)$, likelihood')
ax.fill_between(x1, f_x1_given_C1,alpha = 0.2,color = '#0099FF')

# marginal (evidence)
ax.plot(x1, f_x1, color = '#00448A',label = '$f_{X1}(x_1)$, evidence (marginal)')

# joint
ax.plot(x1, f_x1_joint_C1, color = '#92D050', label = '$f_{X1,Y}(x_1,C_1)$, joint')
ax.fill_between(x1,f_x1_joint_C1, edgecolor = 'k', hatch='///', facecolor="none")

ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])

ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()
plt.close()


## C2
# 图 10. 比较后验概率 fY|X(C2 | x)、似然概率 fX|Y(x|C2)、证据因子 fX(x)、联合概率 fX,Y(x,C2)
fig, ax = plt.subplots()

# posterior
ax.plot(x1, f_C2_given_x1,color = 'r',label = '$f_{Y|X1}(C_2|x_1)$, posterior')

# likelihood
ax.plot(x1, f_x1_given_C2,color = '#0099FF',label = '$f_{X1|Y}(x_1|C_2)$, likelihood')
ax.fill_between(x1, f_x1_given_C2, alpha = 0.2,color = '#0099FF')

# marginal (evidence)
ax.plot(x1, f_x1, color = '#00448A',label = '$f_{X1}(x_1)$, evidence (marginal)')

# joint
ax.plot(x1, f_x1_joint_C2, color = '#92D050', label = '$f_{X1,Y}(x_1,C_2)$, joint')
ax.fill_between(x1,f_x1_joint_C2, edgecolor = 'k', hatch='///', facecolor="none")

ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])

ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()
plt.close()

## C3
# 图 11. 比较后验概率 fY|X(C3 | x)、似然概率 fX|Y(x|C3)、证据因子 fX(x)、联合概率 fX,Y(x,C3
fig, ax = plt.subplots()

# posterior
ax.plot(x1, f_C3_given_x1,color = 'r',label = '$f_{Y|X1}(C_3|x_1)$, posterior')

# likelihood
ax.plot(x1, f_x1_given_C3,color = '#0099FF',label = '$f_{X1|Y}(x_1|C_3)$, likelihood')
ax.fill_between(x1, f_x1_given_C3, alpha = 0.2,color = '#0099FF')

# marginal (evidence)
ax.plot(x1, f_x1, color = '#00448A', label = '$f_{X1}(x_1)$, evidence (marginal)')

# joint
ax.plot(x1, f_x1_joint_C3, color = '#92D050', label = '$f_{X1,Y}(x_1,C_3)$, joint')
ax.fill_between(x1,f_x1_joint_C3, edgecolor = 'k', hatch='///', facecolor="none")

ax.set_ylim([0,1.5])
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_xlim([4,8])

ax.set_xlabel('Sepal length, $x_1$')
ax.legend()
plt.show()
plt.close()































































































































































































































































































