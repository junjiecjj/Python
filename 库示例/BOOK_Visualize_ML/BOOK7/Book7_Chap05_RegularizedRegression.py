#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 多项式回归 + 岭回归正则化



# 导入包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# 生成随机数据
np.random.seed(0)
num = 30

X = np.random.uniform(0,4,num)
y = np.sin(0.4*np.pi * X) + 0.4 * np.random.randn(num)
data = np.column_stack([X,y])


x_array = np.linspace(0,4,101).reshape(-1,1)
degree = 8 # 多项式回归次数
# 将数据扩展为9列
poly = PolynomialFeatures(degree = degree)
X_poly = poly.fit_transform(X.reshape(-1, 1))


fig, axes = plt.subplots(3,2,figsize=(10,20))
axes = axes.flatten()
# 惩罚因子
alpha_array = [0.00001, 0.0001, 0.01, 1, 10, 100]

for ax, alpha_idx in zip(axes,alpha_array):
    # 训练岭回归模型
    ridge = Ridge(alpha=alpha_idx)
    ridge.fit(X_poly, y.reshape(-1,1))
    # 预测
    y_array_pred = ridge.predict(poly.fit_transform(x_array))
    y_poly_pred  = ridge.predict(X_poly)
    data_ = np.column_stack([X,y_poly_pred])
    # 绘制散点图
    ax.scatter(X, y, s=20)
    ax.scatter(X, y_poly_pred, marker = 'x', color='k')
    # 绘制残差
    ax.plot(([i for (i,j) in data_], [i for (i,j) in data]),
            ([j for (i,j) in data_], [j for (i,j) in data]),
             c=[0.6,0.6,0.6], alpha = 0.5)

    ax.plot(x_array, y_array_pred, color='r')
    ax.set_title('Alpha = %f' % alpha_idx)

    # 提取参数
    coef = ridge.coef_[0]; # print(coef)
    intercept = ridge.intercept_[0]; # print(intercept)
    # 回归解析式
    equation = '$y = {:.3f}'.format(intercept)
    for j in range(1, len(coef)):
        equation += ' + {:.3f}x^{}'.format(coef[j], j)
    equation += '$'
    equation = equation.replace("+ -", "-")
    print(equation)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0,4); ax.set_ylim(-2,2); ax.grid(False)




# 多项式回归模型参数随惩罚因子α变化
alphas = np.logspace(4, -2, 100)
degrees = ['Degree = ' + str(d_i) for d_i in range(10)]
colors = plt.cm.jet(np.linspace(0,1,len(degrees)))

coefs = []
for alpha_idx in alphas:
    ridge = Ridge(alpha=alpha_idx)
    ridge.fit(X_poly, y.reshape(-1,1))
    coefs.append(ridge.coef_[0])
coefs = np.array(coefs)


fig, ax = plt.subplots(figsize=(5,3))
for idx in range(9):
    ax.plot(alphas, coefs[:,idx], color = colors[idx])
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1]) # 调转横轴
ax.set_xlabel(r"Regularization strength, penalty $\alpha$")
ax.set_ylabel("Coefficients")
ax.legend(degrees,loc='center left', bbox_to_anchor=(1, 0.5))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 岭回归
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import yfinance as yf
import seaborn as sns
import pandas as pd

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5


tickers = ['^GSPC','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];
labels = ['SP500','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];
stock_levels_df = pd.read_pickle('stock_levels_df.pkl')


# stock_levels_df = yf.download(tickers, start='2020-07-01', end='2020-12-31')
# stock_levels_df.to_csv('stock_levels_df.cvs')
# stock_levels_df.to_pickle('stock_levels_df.pkl')


y_X_df = stock_levels_df['Adj Close'].pct_change()
y_X_df.dropna(inplace = True)

y_X_df.rename(columns={"^GSPC": "SP500"},inplace = True)
X_df = y_X_df[tickers[1:]]
y_df = y_X_df[['SP500']]



# OLS
import statsmodels.api as sm

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())
b = model.fit().params
b = b.values

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  图 8. 随着 α 增大，岭回归参数变化
# Ridge regression
clf = Ridge()
coefs = []
errors = []
coeff_df = pd.DataFrame()

alphas = np.logspace(-4, 2, 200)

# Train the model with different regularisation strengths
for alpha_i in alphas:
    clf.set_params(alpha=alpha_i)
    clf.fit(X_df, y_df)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(clf.coef_, b.reshape(1,-1)))

    b_i = clf.coef_ #  (1, 13)
    b_X_df = pd.DataFrame(data = b_i[:,1:].T, index = tickers[1:], columns=[alpha_i])

    coeff_df = pd.concat([coeff_df, b_X_df], axis = 1) # (12, 200)

fig, ax = plt.subplots(figsize = (8,5))
h = sns.lineplot(data=coeff_df.T, markers=False, dashes=False,palette = "husl")
h.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.axhline(y=0, color='k', linestyle='--')
ax.set_xscale('log')
plt.tight_layout()
plt.grid(b=True, which='minor', color='0.8')
# ax.grid(which='minor', axis='x', linestyle='--')


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 图 9. 和 OLS 相比，岭回归参数误差
fig, ax = plt.subplots(figsize = (8,5))

ax.plot(alphas, errors)
plt.fill_between(alphas, errors, color = '#DEEAF6')
ax.set_xscale('log')
plt.xlabel('$\u03B1$')
plt.ylabel('Coefficient error')
plt.axis('tight')
plt.ylim(0,0.015)
plt.grid(b=True, which='minor', color='0.8')
plt.show()


# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  图 16. 随着 α 增大，套索回归参数变化
# # Lasso regression
# clf = Lasso()
# coefs = []
# errors = []
# coeff_df = pd.DataFrame()

# alphas = np.logspace(-4, 2, 200)

# # Train the model with different regularisation strengths
# for alpha_i in alphas:
#     clf.set_params(alpha=alpha_i)
#     clf.fit(X_df, y_df)
#     coefs.append(clf.coef_)
#     errors.append(mean_squared_error(clf.coef_.reshape(1,-1), b.reshape(1,-1)))

#     b_i = clf.coef_.reshape(1,-1)
#     b_X_df = pd.DataFrame(data=b_i[:,1:].T, index = tickers[1:], columns=[alpha_i])

#     coeff_df = pd.concat([coeff_df, b_X_df], axis = 1)

# fig, ax = plt.subplots(figsize = (8,5))
# h = sns.lineplot(data=coeff_df.T, markers=False, dashes=False, palette = "husl")
# plt.axhline(y=0, color='k', linestyle='--')
# h.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# ax.set_xscale('log')
# plt.grid(b=True, which='minor', color='0.8')
# # ax.grid(which='minor', axis='x', linestyle='--')


# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 图 9. 和 OLS 相比，岭回归参数误差
# fig, ax = plt.subplots(figsize = (8,5))

# ax.plot(alphas, errors)
# plt.fill_between(alphas, errors, color = '#DEEAF6')
# ax.set_xscale('log')
# plt.xlabel('$\u03B1$')
# plt.ylabel('Coefficient error')
# plt.axis('tight')
# plt.ylim(0,0.015)
# plt.grid(b=True, which='minor', color='0.8')
# plt.show()









































































































































































































































































































































