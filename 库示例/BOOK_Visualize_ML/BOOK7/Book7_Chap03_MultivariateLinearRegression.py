


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 二元OLS线性回归
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import yfinance as yf

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = False
p["xtick.minor.visible"] = False
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5


# y_X_df = yf.download(['AAPL','MCD','^GSPC'], start='2020-01-01', end='2020-12-31')
# y_X_df.to_pickle('y_X_df.pkl')
# or
y_X_df = pd.read_pickle('y_X_df.pkl')

y_X_df = y_X_df['Adj Close'].pct_change()
y_X_df.dropna(inplace = True)
y_X_df.rename(columns={"^GSPC": "SP500"}, inplace = True)

X_df = y_X_df[['AAPL','MCD']]
y_df = y_X_df[['SP500']]


#>>>>>>>>>>>>>>>>>>>>>>  Data analysis
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X_df["AAPL"], X_df["MCD"], y_df, s = 8, alpha = 0.5)

ax.set_xlabel('AAPL')
ax.set_ylabel('MCD')
ax.set_zlabel('SP500')
ax.set_proj_type('ortho')
ax.set_xlim([-0.15,0.15])
ax.set_ylim([-0.15,0.15])
ax.set_zlim([-0.15,0.15])
# plt.savefig('三维散点数据.svg')

#>>>>>>>>>>>>>>>>>>>>>>
g = sns.pairplot(y_X_df)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="viridis_r")
g.map_diag(sns.histplot, kde=False, color = 'b')


#>>>>>>>>>>>>>>>>>>>>>> # covariance matrix

SIGMA = y_X_df.cov()
fig, axs = plt.subplots()
h = sns.heatmap(SIGMA, annot=True,cmap='RdBu_r')
h.set_aspect("equal")

#>>>>>>>>>>>>>>>>>>>>>>   correlation matrix
RHO = y_X_df.corr()
fig, axs = plt.subplots()
h = sns.heatmap(RHO, annot=True,cmap='RdBu_r')
h.set_aspect("equal")

#>>>>>>>>>>>>>>>>>>>>>>   Volatility vector space
Angles = np.arccos(RHO)*180/np.pi
fig, axs = plt.subplots()

h = sns.heatmap(Angles, annot=True,cmap='RdBu_r')
h.set_aspect("equal")

#>>>>>>>>>>>>>>>>>>>>>>   Regression
# add a column of ones
X_df = sm.add_constant(X_df)
model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())
p = model.fit().params
print(p)

#>>>>>>>>>>>>>>>>>>>>>>   generate x-values for your regression line (two is sufficient)
xx1,xx2 = np.meshgrid(np.linspace(-0.15,0.15,20), np.linspace(-0.15,0.15,20))
yy = p.AAPL*xx1 + p.MCD*xx2 + p.const

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X_df["AAPL"], X_df["MCD"], y_df, s = 8, alpha = 0.5)
ax.plot_wireframe(xx1, xx2, yy, color = '0.5')
ax.set_xlim([-0.15,0.15])
ax.set_ylim([-0.15,0.15])
ax.set_zlim([-0.15,0.15])
ax.set_xlabel('AAPL')
ax.set_ylabel('MCD')
ax.set_zlabel('SP500')
ax.set_proj_type('ortho')
# plt.savefig('三维散点数据 + 回归平面.svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 高斯条件概率
# initializations
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yfinance as yf

tickers = ['^GSPC','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];
labels = ['SP500','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];
# stock_levels_df = yf.download(tickers, start='2020-07-01', end='2020-12-31')
# stock_levels_df.to_csv('stock_levels_df.csv')
# stock_levels_df.to_pickle('stock_levels_df.pkl')
stock_levels_df = pd.read_pickle('stock_levels_df.pkl')


y_X_df = stock_levels_df['Adj Close'].pct_change()
y_X_df.dropna(inplace = True)
y_X_df.rename(columns={"^GSPC": "SP500"},inplace = True)

X_df_no_1 = y_X_df[tickers[1:]]
y_df = y_X_df[['SP500']]



#%% Regression

import statsmodels.api as sm

# add a column of ones
X_df = sm.add_constant(X_df_no_1)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

#%% Regression analysis

y = np.array(y_df.values) # (126, 1)
X = X_df.values # (126, 13)

H = X@np.linalg.inv(X.T@X)@X.T # (126, 126)
# coefficients
b = np.linalg.inv(X.T@X)@X.T@y # (13, 1)

y_hat = H@y   # (126, 1)
e = y - y_hat # (126, 1)

#%% Analysis of Variance
n = y.shape[0] # 126
k = X.shape[1] # 13
D = k - 1 # 12

I = np.identity(n)
J = np.ones((n,n))
vec_1 = np.ones_like(y)

y_bar = vec_1.T@y/n

# Sum of Squares for Total, SST
SST = y.T@(I - J/n)@y
MST = SST/(n - 1)
MST = MST[0,0]

#%% Sum of Squares for Error, SSE
SSE = y.T@(I - H)@y

# mean squared error, MSE
MSE = SSE/(n - k) # array([[2.74911771e-05]])
MSE_ = e.T@e/(n - k)
MSE = MSE[0,0] # 2.7491177108143528e-05

#%% Sum of Squares for Regression, SSR
SSR = y.T@(H - J/n)@y
MSR = SSR/D
MSR = MSR[0,0]

#%% 正交关系 Orthogonal relationships
# 第一个直角三角形
print('SST = ',SST)
print('SSR + SSE = ',SSR + SSE)
# SST =  [[0.01374375]]
# SSR + SSE =  [[0.01374375]]
print('================')
# 第二个直角三角形
print('y.T@y = ',y.T@y)
print('y_hat.T@y_hat + e.T@e = ',y_hat.T@y_hat + e.T@e)
# y.T@y =  [[0.01402271]]
# y_hat.T@y_hat + e.T@e =  [[0.01402271]]

print('e.T@(y_hat - y_bar*vec_1) = ', e.T@(y_hat - y_bar))
# e.T@(y_hat - y_bar*vec_1) =  [[2.99235436e-18]]

print('================')
# 第三个直角三角形
print('(y_bar*vec_1).T @ (y - y_bar*vec_1) = ', (y_bar*vec_1).T @ (y - y_bar*vec_1) )\
# array([[5.4385737e-21]])

print('================')
# 第四个直角三角形
print('e.T@vec_1 = ', e.T@vec_1)
# e.T@vec_1 =  [[-6.9388939e-18]]

print('================')
print('e.T@(y - y_bar*vec_1) = ', e.T@(y - y_bar))
# e.T@(y - y_bar*vec_1) =  [[0.0031065]]
print('================')
print('e.T@X = ', e.T@X)

print('================')
print('e.T@X@b = ', e.T@X@b)
# e.T@X@b =  [[2.92208136e-18]]
#%% R squared goodness of fit
R_squared = SSR/SST
R_sqaured_adj = 1 - MSE/MST

#%% F test
F = MSR/MSE
from scipy import stats
p_value_F = 1.0 - stats.f.cdf(F,k - 1,n - k)

#%% Log-likelihood
sigma_MLE = np.sqrt(SSE/n)

ln_L = -n*np.log(sigma_MLE*np.sqrt(2*np.pi)) - SSE/2/sigma_MLE**2

AIC = 2*k - 2*ln_L
BIC = k*np.log(n) - 2*ln_L

#%% t 检验
C = MSE*np.linalg.inv(X.T@X)

SE_b = np.sqrt(np.diag(C))
SE_b = np.matrix(SE_b).T

T = b/SE_b
p_one_side = 1 - stats.t(n - k).cdf(np.abs(T))
p = p_one_side*2
# P > |t|

#%% confidence interval of coefficients, 95%
alpha = 0.05
t = stats.t(n - k).ppf(1 - alpha/2)
b_lower_CI = b - t*SE_b # 0.025
b_upper_CI = b + t*SE_b # 0.975
#%% 多重共线性

print('Rank of X')
print(np.linalg.matrix_rank(X))

print('det(X.T@X)')
print(np.linalg.det(X.T@X))

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

VIF_X_df = pd.Series([VIF(X_df.values, i) for i in range(X_df.shape[1])], index=X_df.columns)

VIF_X_no_1_df = pd.Series([VIF(X_df_no_1.values, i) for i in range(X_df_no_1.shape[1])], index=X_df_no_1.columns)

print(VIF_X_no_1_df)
print(VIF_X_df)


#%% Conditional probability

#>>>>>>>>>>>>>>>>>>>>>>>>>  covariance matrix
SIGMA_df = y_X_df.cov()
SIGMA = SIGMA_df.to_numpy() #  (13, 13)

fig, axs = plt.subplots()
h = sns.heatmap(SIGMA_df,cmap='RdBu_r', linewidths=.05)
h.set_aspect("equal")
h.set_title('$\Sigma$')

# blocks
SIGMA_Xy = np.matrix(SIGMA[1:,0]).T # (12, 1)
SIGMA_XX = np.matrix(SIGMA[1:,1:]) # (12, 12)
SIGMA_XX_inv = np.linalg.inv(SIGMA_XX)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(axs[0])
ax = sns.heatmap(SIGMA_XX,cmap='RdBu_r', cbar=False, xticklabels = labels[1:], yticklabels = labels[1:], linewidths=.05)
ax.set_aspect("equal")
plt.title('$\Sigma_{XX}$')

plt.sca(axs[1])
ax = sns.heatmap(SIGMA_XX_inv,cmap='RdBu_r', cbar=False, xticklabels = labels[1:], yticklabels = labels[1:], linewidths=.05)
ax.set_aspect("equal")
plt.title('$\Sigma_{XX}^{-1}$')
#>>>>>>>>>>>>>>>>>>>>>>>>>
b = SIGMA_XX_inv@SIGMA_Xy # (12, 1)

fig, axs = plt.subplots(1, 5, figsize=(12, 3))
plt.sca(axs[0])
ax = sns.heatmap(b,cmap='RdBu_r', cbar=False, yticklabels = labels[1:], linewidths=.05)
ax.set_aspect("equal")
plt.title('$b$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(SIGMA_XX_inv,cmap='RdBu_r', cbar=False, xticklabels = labels[1:], yticklabels = labels[1:], linewidths=.05)
ax.set_aspect("equal")
plt.title('$\Sigma_{XX}^{-1}$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(SIGMA_Xy,cmap='RdBu_r', cbar=False, xticklabels = [labels[0]], yticklabels = labels[1:], linewidths=.05)
ax.set_aspect("equal")
plt.title('$\Sigma_{Xy}$')

#%% calculate coefficient, b0
MU = y_X_df.mean()
MU = np.matrix(MU.to_numpy())

b0 = MU[0,0] - MU[0,1:]@b

fig, axs = plt.subplots(1, 7, figsize=(12, 3))
plt.sca(axs[0])
ax = sns.heatmap(b0,cmap='RdBu_r', cbar=False, linewidths=.05,xticklabels = [], yticklabels = [])
ax.set_aspect("equal")
plt.title('$b_0$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(np.matrix(MU[0,0]),cmap='RdBu_r', cbar=False, linewidths=.05, xticklabels = [], yticklabels = [])
ax.set_aspect("equal")
plt.title('$\mu_{y}$')

plt.sca(axs[3])
plt.title('-')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(MU[0,1:],cmap='RdBu_r', cbar=False, xticklabels = labels[1:], yticklabels = [], linewidths=.05)
ax.set_aspect("equal")
plt.title('$\mu_{X}$')


plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(b,cmap='RdBu_r', cbar=False, yticklabels = labels[1:], xticklabels = [], linewidths=.05)
ax.set_aspect("equal")
plt.title('$b$')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 方差分析，多元线性回归实例

# initializations
import pandas as pd
# import pandas_datareader as web
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


tickers = ['^GSPC','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];

# stock_levels_df = yf.download(tickers, start='2020-07-01', end='2020-12-31')
# stock_levels_df.to_csv('stock_levels_df.csv')
# stock_levels_df.to_pickle('stock_levels_df.pkl')

stock_levels_df = pd.read_pickle('stock_levels_df.pkl')

y_X_df = stock_levels_df['Adj Close'].pct_change()
y_X_df.dropna(inplace = True)

y_X_df.rename(columns={"^GSPC": "SP500"},inplace = True)
X_df = y_X_df[tickers[1:]]
y_df = y_X_df[['SP500']]

labels = ['SP500','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];

#%% Lineplot of stock prices
# 图 14. 股价数据，起始值归一化
normalized_stock_levels = stock_levels_df['Adj Close']/stock_levels_df['Adj Close'].iloc[0]

g = sns.relplot(data=normalized_stock_levels, dashes = False, kind="line") # , palette="coolwarm"
g.set_xlabels('Date')
g.set_ylabels('Normalized closing price')
g.set_xticklabels(rotation=45)

# 图 15. [y, X] 日收益率热图
fig, ax = plt.subplots()
ax = sns.heatmap(y_X_df, cmap='RdBu_r', cbar_kws={"orientation": "vertical"}, yticklabels=False, vmin = -0.2, vmax = 0.2)
plt.title('[y, X]')

#%% 图 16. [y, X] 数据协方差矩阵
SIGMA = y_X_df.cov()
fig, axs = plt.subplots()
h = sns.heatmap(SIGMA,cmap='RdBu_r', linewidths=.05)
h.set_aspect("equal")

# 图 17. 日波动率柱状图
vols = np.sqrt(np.diag(SIGMA))
fig, ax = plt.subplots()
plt.bar(labels,vols)
plt.xticks(rotation = 45)
# Rotates X-Axis Ticks by 45-degrees
plt.ylabel('Daily volatility (standard deviation)')

#%% 图 18. [y, X] 数据相关性系数矩阵热图

fig, ax = plt.subplots()
# Compute the correlation matrix
RHO = y_X_df.corr()

h = sns.heatmap(RHO, cmap="RdBu_r", square=True, linewidths=.05, annot=False)
h.set_aspect("equal")

# 图 19. 股价收益率和 S&P 500 收益率相关性系数柱状图
fig, ax = plt.subplots()
plt.bar(labels,RHO['SP500'].iloc[:].values)
plt.xticks(rotation = 45)
# Rotates X-Axis Ticks by 45-degrees
plt.ylabel('Correlation with S&P 500')
RHO.to_excel('corr.xlsx')

#%% 图 20. [y, X] 标准差向量夹角矩阵热图，余弦相似性

Angles = np.arccos(RHO)*180/np.pi
fig, axs = plt.subplots()

h = sns.heatmap(Angles, annot=False,cmap='RdBu_r', vmin = 30, vmax = 115)
h.set_aspect("equal")
Angles.to_excel('output.xlsx')
#%% Regression

import statsmodels.api as sm
# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

p = model.fit().params
print(p)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

















#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



































































































































































































































































































































































































































































































































































































