



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk7_Ch02_01


# 一元OLS线性回归

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yfinance as yf
import statsmodels.api as sm


p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = False
p["xtick.minor.visible"] = False
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5



# # 下载股价数据
# y_x_df_raw = yf.download(['AAPL','GSPC'], start='2020-01-01', end='2020-12-31')
# y_x_df_raw.head()
# y_x_df_raw.to_csv('y_x_df_raw.csv')
# y_x_df_raw.to_pickle('y_x_df_raw.pkl')

# 如果不能下载，就用pandas.read_csv() 或 pandas.read_pickle() 读入
# 建议使用 pandas.read_pickle()
# y_x_df_raw = pd.read_csv('y_x_df_raw.csv')
y_x_df_raw = pd.read_pickle('y_x_df_raw.pkl')

# 计算收益率
# 收盘价，计算日回报率
y_x_df = y_x_df_raw['Adj Close'].pct_change()
y_x_df.dropna(inplace = True)


y_x_df.rename(columns={"^GSPC": "SP500"},inplace = True)

y_x_df.head()


x_df = y_x_df[['SP500']]
y_df = y_x_df[['AAPL']]


# 数据分析
#%% Data analysis
sns.jointplot(data=y_x_df, x="SP500", y="AAPL", kind = 'scatter', xlim = [-0.15,0.15],ylim = [-0.15,0.15])

# marginal and joint KDE plots
sns.jointplot(data=y_x_df, x="SP500", y="AAPL", kind="kde", cmap = 'Blues', fill = True, xlim = [-0.15,0.15],ylim = [-0.15,0.15])


# marginal and joint KDE plots
# sns.jointplot(data=y_x_df, x="SP500", y="AAPL", kind="reg", xlim = [-0.15,0.15],ylim = [-0.15,0.15])


# 协方差矩阵
SIGMA = y_x_df.cov()

fig, axs = plt.subplots()
h = sns.heatmap(SIGMA, annot=True,cmap='RdBu_r')
h.set_aspect("equal")
print(np.sqrt(np.diag(SIGMA)))


# 相关系系数矩阵
RHO = y_x_df.corr()
fig, axs = plt.subplots()
h = sns.heatmap(RHO, annot=True,cmap='RdBu_r')
h.set_aspect("equal")

# 相关系转换为角度
Angles = np.arccos(RHO)*180/np.pi
fig, axs = plt.subplots()
h = sns.heatmap(Angles, annot=True,cmap='RdBu_r')
h.set_aspect("equal")


# 向量
def draw_vector(vector, RBG):
    array = np.array([[0, 0, vector[0], vector[1]]])
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1,color = RBG)

angle = Angles['AAPL']['SP500']*np.pi/180

vols = np.sqrt(np.diag(SIGMA))
v_1_x = vols[1]
v_1_y = 0

v_2_x = vols[0]*np.cos(angle)
v_2_y = vols[0]*np.sin(angle)

fig, ax = plt.subplots()
draw_vector([v_1_x,v_1_y], np.array([0,112,192])/255)
draw_vector([v_2_x,v_2_y], np.array([255,0,0])/255)
plt.ylabel('$y, TSLA$')
plt.xlabel('$x, S&P500$')
plt.axis('scaled')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
ax.set_xlim([-0.01, 0.03])
ax.set_ylim([-0.01, 0.03])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])


# 无截距一元线性回归
model_no_intercept = sm.OLS(y_df, x_df)
results_no_intercept = model_no_intercept.fit()
print(results_no_intercept.summary())



# OLS线性回归
# 增加一列全1
X_df = sm.add_constant(x_df)
model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

# 可视化
p = model.fit().params
# generate x-values for  regression line
x = np.linspace(x_df.min(),x_df.max(),10)

fig, ax = plt.subplots()
# scatter-plot data
plt.scatter(x_df, y_df, alpha = 0.5)
plt.plot(x, p.const + p.SP500 * x,color = 'r')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.axis('scaled')
plt.ylabel('AAPL daily log return')
plt.xlabel('S&P 500 daily log return, market')
plt.xlim([-0.15,0.15])
plt.ylim([-0.15,0.15])

sns.jointplot(x=x_df['SP500'], y=y_df['AAPL'], kind="reg", xlim = [-0.15,0.15],ylim = [-0.15,0.15])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 方差分析

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import yfinance as yf




y_x_df_raw = pd.read_pickle('y_x_df_raw.pkl')

# 计算收益率
# 收盘价，计算日回报率
y_x_df = y_x_df_raw['Adj Close'].pct_change()
y_x_df.dropna(inplace = True)
y_x_df.rename(columns={"^GSPC": "SP500"},inplace = True)
x_df = y_x_df[['SP500']]
y_df = y_x_df[['AAPL']]

# 线性回归
# 增加一列全1
X_df = sm.add_constant(x_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

# ANOVA表格
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

data = pd.DataFrame({'x': x_df['SP500'], 'y': y_df['AAPL']})
model_V2 = ols("y ~ x", data).fit()
anova_results = anova_lm(model_V2, typ=1)
print(anova_results)


# 总离差平方和 (Sum of Squares for Total, SST)，也称 TSS (total sum of squares)。
y_mean = y_df.mean()
# Sum of Squares for Total, SST
SST = ((y_df - y_mean)**2).sum()
print(SST)


# 总离差自由度 DFT
n = len(y_df)
print(n)
DFT = n - 1 # 250
# mean square total, MST
MST = SST/DFT
print(MST)
type(results)

# 拟合后预测值
y_hat = results.fittedvalues
y_hat = y_hat.to_frame()
y_hat = y_hat.rename(columns={0: 'AAPL'})
# y_hat.mean() == y_df.mean()

# 回归平方和 (Sum of Squares for Regression, SSR)，也称 ESS (explained sum of squares)。
SSR = ((y_hat - y_mean)**2).sum()
print(SSR)
# 回归自由度 (degrees of freedom for regression model, DFR)
DFR = 1
# 平均回归平方 (mean square regression, MSR)
MSR = SSR/DFR
print(MSR)


# 残差平方和 (Sum of Squares for Error, SSE)，也称 RSS (residual sum of squares)。
SSE = ((y_df - y_hat)**2).sum()
print(SSE)
# 残差自由度 (degrees of freedom for error, DFE)
DFE = n - DFR - 1  # 249
# 残差平均值 (mean squared error, MSE)
MSE = SSE/DFE
print(MSE)

# 拟合优度
# 决定系数 (coefficient of determination，R2)
R2 = SSR/SST

# 计算修正决定系数
R2_adj = 1 - MSE/MST


# 计算F检验的统计量
F_test = MSR/MSE
print(F_test)

# 验算F检验的统计量
N = results.nobs
k = results.df_model+1
dfm, dfe = k-1, N - k
F = results.mse_model / results.mse_resid
print(F)



import scipy.stats as stats
p = 1.0 - stats.f.cdf(F,dfm,dfe)

alpha = 0.01
# F = 549.7
# n = 252
# D = 1
# p = D + 1

fdistribution = stats.f(p - 1, n - p)
# build an F-distribution object
f_critical = fdistribution.ppf(1 - alpha)

p_value = 1 - stats.f.cdf(F, p - 1, n - p)



x_points = x_df.values.T
y_points = y_df.values.T
y_hat_points = y_hat.values.T

p = model.fit().params

# generate x-values for  regression line
x = np.linspace(x_df.min(),x_df.max(),10)

# 可视化
fig, ax = plt.subplots()
plt.scatter(x_df, y_df, alpha = 0.5)
plt.plot(x_points, y_hat_points,'+k');

plt.plot(x, p.const + p.SP500 * x,color = 'r')
plt.plot(np.vstack((x_points,x_points)), np.vstack((y_points,y_hat_points)), color = [0.7,0.7,0.7], zorder = 1);

plt.axis('scaled')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim([-0.15,0.15])
plt.ylim([-0.15,0.15])




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk7_Ch02_03 回归分析

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yfinance as yf
import statsmodels.api as sm



p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = False
p["xtick.minor.visible"] = False
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5


# 下载、处理数据
y_x_df = yf.download(['AAPL','^GSPC'], start='2020-01-01', end='2020-12-31')
y_x_df = y_x_df['Adj Close'].pct_change()
y_x_df.dropna(inplace = True)

y_x_df.rename(columns={"^GSPC": "SP500"},inplace = True)
y_x_df.to_pickle('y_x_df.pkl')
# 如果不能下载数据，请用pandas.read_pickle() 从配套文件读入数据
x_df = y_x_df[['SP500']]
y_df = y_x_df[['AAPL']]


# 线性回归
# add a column of ones
X_df = sm.add_constant(x_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

model_params = model.fit().params

x_mean = x_df.mean().values
y_mean = y_df.mean().values

n = len(y_df)

# predicted
y_hat = results.fittedvalues

y_hat = y_hat.to_frame()
y_hat = y_hat.rename(columns={0: 'AAPL'})

DFR = 1


# Sum of Squares for Error, SSE
SSE = ((y_df - y_hat)**2).sum()

# degrees of freedom for error, DFE
DFE = n - DFR - 1

# t-test
MSE = SSE/DFE
MSE = MSE.values
# 计算MSE

b1 = model_params.SP500
# 斜率系数

SSD_x = np.sum((x_df.values - x_mean)**2)

SE_b1 = np.sqrt(MSE/SSD_x)
# 标准误

T_b1 = (b1 - 0)/SE_b1
# b1的t检验统计量
print(T_b1)


b0 = model_params.const
# 截距系数

SE_b0 = np.sqrt(MSE*(1/n + x_mean**2/SSD_x))
# 标准误

T_b0 = (b0 - 0)/SE_b0
# b0的t检验统计量

print(T_b0)

from scipy import stats

pval_b1 = stats.t.sf(np.abs(T_b1), n-2)*2
print(pval_b1)
pval_b0 = stats.t.sf(np.abs(T_b0), n-2)*2
print(pval_b0)


# confidence intervals of coefficients
alpha = 0.05
# 显著水平
t_95 = stats.t.ppf(1 - alpha/2, DFE)
# t值

# 系数b1的1 – α 置信区间
b1_upper_95 = b1 + t_95*SE_b1
print(b1_upper_95)
b1_lower_95 = b1 - t_95*SE_b1
print(b1_lower_95)

# 系数b0的1 – α 置信区间
b0_upper_95 = b0 + t_95*SE_b0
print(b0_upper_95)
b0_lower_95 = b0 - t_95*SE_b0
print(b0_lower_95)

# 可视化
# generate x-values for regression line
x_i = np.linspace(-0.15,0.15,50)

# predicted values
y_i = b0 + b1* x_i

alpha = 0.05
t_95 = stats.t.ppf(1 - alpha/2, DFE)

CI_95 = t_95 * np.sqrt(MSE) * np.sqrt(1/n + (x_i - x_mean)**2 / np.sum((x_df.values - x_mean)**2))

CI_upper_95 = y_i + CI_95
CI_lower_95 = y_i - CI_95


alpha = 0.01
t_99 = stats.t.ppf(1 - alpha/2, DFE)

CI_99 = t_99 * np.sqrt(MSE) * np.sqrt(1/n + (x_i - x_mean)**2 / np.sum((x_df.values - x_mean)**2))

CI_upper_99 = y_i + CI_99
CI_lower_99 = y_i - CI_99


# confidence band
fig, ax = plt.subplots()
plt.scatter(x_df, y_df, alpha = 0.5)

plt.plot(x_i, y_i,color = 'r')

# plot confidence interval
ax.fill_between(x_i,CI_lower_95, CI_upper_95, color = 'b', alpha = 0.5)
ax.fill_between(x_i,CI_lower_99, CI_upper_99, color = 'b', alpha = 0.25)

plt.axis('scaled')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim([-0.15,0.15])
plt.ylim([-0.15,0.15])

y_x_df.head()


# linear regression with confidence interval, Seaborn

sns.regplot(y_x_df, x = 'SP500', y = 'AAPL', ci=95)
plt.axis('scaled')


sns.regplot(y_x_df, x = 'SP500', y = 'AAPL', ci=99)
plt.axis('scaled')


alpha = 0.05
t_95 = stats.t.ppf(1 - alpha/2, DFE)

pi_95 = t_95 * np.sqrt(MSE) * np.sqrt(1 + 1/n + (x_i - x_mean)**2 / np.sum((x_df.values - x_mean)**2))

pi_upper_95 = y_i + pi_95
pi_lower_95 = y_i - pi_95

alpha = 0.01
t_99 = stats.t.ppf(1 - alpha/2, DFE)

pi_99 = t_99 * np.sqrt(MSE) * np.sqrt(1 + 1/n + (x_i - x_mean)**2 / np.sum((x_df.values - x_mean)**2))

pi_upper_99 = y_i + pi_99
pi_lower_99 = y_i - pi_99



# plot predicting interval
fig, ax = plt.subplots()
plt.scatter(x_df,y_df, zorder = 10,
            color = 'b', alpha = 0.25);

plt.plot(x_i, y_i,color = 'r')


ax.fill_between(x_i,pi_lower_95, pi_upper_95, color = 'b', alpha = 0.25)
ax.fill_between(x_i,pi_lower_99, pi_upper_99, color = 'b', alpha = 0.25)

plt.axis('scaled')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim([-0.15,0.15])
plt.ylim([-0.15,0.15])


# Log-Likelihood Function
# SSE = SSE.values
s2 = SSE / n

# maximum likelihood estimator of error variance
# Log Likelihood function
log_L = n*(-np.log(np.sqrt(s2*2*np.pi))) - n/2
print(log_L)


AIC = 2*(DFR + 1) - 2*log_L
print(AIC)
BIC = (DFR + 1)*np.log(n) - 2*log_L
print(BIC)

# Residual analysis
e_df = y_df - y_hat;


# confidence band
fig, ax = plt.subplots()
plt.scatter(x_df,e_df, alpha = 0.5);
plt.axhline(y=0, color='r', linestyle='--')
plt.axis('scaled')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim([-0.15,0.15])
plt.ylim([-0.15,0.15])


# skewness
S = np.mean(e_df**3)/np.mean(e_df**2)**(3/2)
print(S)
S_2 = stats.skew(e_df,bias = True)
print(S_2)

## kurtosis
K = np.mean(e_df**4)/np.mean(e_df**2)**(4/2)
print(K)
K_2 = stats.kurtosis(e_df,fisher=False,bias = True)
print(K_2)



# Omnibus test for normality
(Omnibus_test, p) = stats.normaltest(e_df)
print(Omnibus_test)


fig, ax = plt.subplots()
sns.distplot(e_df)


# Jarque-Bera test
JB = (n/6.0) * ( S**2.0 + (1.0/4.0)*( K - 3.0 )**2.0 )
print(JB)
p_JB = 1.0 - stats.chi2(2).cdf(JB)
print(p_JB)

# Autocorrelation
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

# Durbin-Watson test
DW = np.sum(np.diff(e_df['AAPL'].values)**2)/SSE
print(DW)
plot_acf(e_df, lags=20)
pyplot.show()


# Condition number, multicollinearity
X = np.matrix(X_df)
eigen_values, V = np.linalg.eig(X.T * X)
print(eigen_values)

condition_number = np.sqrt(eigen_values.max()/eigen_values.min())
print(condition_number)
















































































































































































































































































































































































































































