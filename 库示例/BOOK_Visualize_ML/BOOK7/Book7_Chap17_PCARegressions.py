




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 正交回归，一元


# initializations and download results
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm



p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

# y_levels_df = web.get_data_yahoo(['AAPL'], start = '2020-01-01', end = '2020-12-31')

from datetime import datetime
startdate = datetime(2020,1,1)
enddate = datetime(2020,12,31)
# y_levels_df = yf.download(tickers = ['AAPL'], start = startdate, end = enddate)
# y_levels_df.to_pickle('y_levels_df.pkl')

y_levels_df = pd.read_pickle('y_levels_df.pkl')
y_levels_df.round(2).head()
y_df = y_levels_df['Adj Close'].pct_change()
y_df = y_df.dropna()

# x_levels_df = web.get_data_yahoo(['^GSPC'], start = '2020-01-01', end = '2020-12-31')
# x_levels_df = yf.download(tickers = ['^GSPC'], start = startdate, end = enddate)
# x_levels_df.to_pickle('x_levels_df.pkl')
x_levels_df = pd.read_pickle('x_levels_df.pkl')
x_levels_df.round(2).head()
x_df = x_levels_df['Adj Close'].pct_change()
x_df = x_df.dropna()

# x_df = x_df.rename(columns={"^GSPC": "SP500"})
x_y_df = pd.concat([x_df, y_df], axis=1, join="inner")


from scipy import odr

# Define a function to fit the data with
def linear_func(b, x):
   b0, b1 = b
   return b1*x + b0


# Create a model for fitting
linear_model = odr.Model(linear_func)
# Load data to the model
data = odr.RealData(x_df.T, y_df.T)
# Set up ODR with the model and data
odr = odr.ODR(data, linear_model, beta0=[0., 1.])
# Solve the regression
out = odr.run()
# Use the in-built pprint method to display results
out.pprint()
y_df.mean()

#%% TLS, matrix computation
import statsmodels.api as sm
SIMGA = x_y_df.cov()
Lambda, V = np.linalg.eig(SIMGA)

idx = Lambda.argsort()[::-1]
Lambda = Lambda[idx]
V = V[:,idx]

lambda_min = np.min(Lambda)
b1_TLS = -V[0, 1]/V[1, 1]
print(b1_TLS)

b0_TLS = y_df.mean() - b1_TLS*x_df.mean()
print(b0_TLS)

#%% OLS regression
# add a column of ones
X_df = sm.add_constant(x_df)
model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())
p = model.fit().params



#%% visualization
b0 = out.beta[0]
b1 = out.beta[1]

# generate x-values for  regression line
x_ = np.linspace(x_df.min(),x_df.max(),10)
p

fig, ax = plt.subplots()
# scatter-plot data
plt.scatter(x_df, y_df, alpha = 0.5, s = 8,label = 'Data')
plt.plot(x_, p.const + p['Adj Close'] * x_, color = 'r', label = 'OLS')
plt.plot(x_, b0 + b1 * x_, color = 'b', label = 'TLS')
# plt.plot(x_, b0_TLS + b1_TLS * x_, color = 'g', label = 'TLS hand')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.axis('scaled')
plt.legend(loc='lower right')

plt.axis('scaled')
plt.ylabel('AAPL daily return')
plt.xlabel('S&P 500 daily return, market')
plt.xlim([-0.15,0.15])
plt.ylim([-0.15,0.15])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 正交回归，二元

###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis

def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0.25, 0.25, 0.25]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs
if not hasattr(Axis, "_get_coord_info_old"):
    Axis._get_coord_info_old = Axis._get_coord_info
Axis._get_coord_info = _get_coord_info_new
###patch end###

# bi-variate regression

# initializations and download results
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yfinance as yf
import statsmodels.api as sm

# X_y_df = yf.download(['AAPL','MCD','^GSPC'], start='2020-01-01', end='2020-12-31')
# X_y_df.to_pickle('X_y_df.pkl')
X_y_df = pd.read_pickle('X_y_df.pkl')
X_y_df = X_y_df['Adj Close'].pct_change()
X_y_df.dropna(inplace = True)


X_y_df.rename(columns={"^GSPC": "SP500"},inplace = True)
X_df = X_y_df[['AAPL','MCD']]
y_df = X_y_df[['SP500']]



#%% USE ODR in scipy

from scipy.odr import *

# Define a function to fit data
def linear_func(b, x):
   # b0, b1, b2 = b
   # x1, x2 = x
   # return b2*x2 + b1*x1 + b0
   b0 = b[0]
   b_ = b[1:]
   return b_.T@x + b0

# Create a model for fitting
linear_model = Model(linear_func)
# Create a RealData object using our initiated data
data = RealData(X_df.T, y_df.T)
# Set up ODR with the model and data
odr = ODR(data, linear_model, beta0=[0., 1., 1])
# Run the regression
out = odr.run()
# Use pprint method to display results
out.pprint()

b0_TLS = out.beta[0]
b1_TLS = out.beta[1]
b2_TLS = out.beta[2]


#%% TLS, matrix computation

SIMGA = X_y_df.cov()

Lambda, V = np.linalg.eig(SIMGA)

idx = Lambda.argsort()[::-1]
Lambda = Lambda[idx]
V = V[:,idx]

lambda_min = np.min(Lambda)

b1_TLS_ = -V[0,2]/V[2,2] # 公式(39)
b2_TLS_ = -V[1,2]/V[2,2] # 公式(39)

print(b1_TLS_)
print(b2_TLS_)
b0_TLS_ = y_df.mean().values - [b1_TLS_, b2_TLS_]@X_df.mean().values
print(b0_TLS_)




#%% OLS Regression
# add a column of ones
X_df = sm.add_constant(X_df)
model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

p = model.fit().params
print(p)

yy_OLS = p.AAPL*xx1 + p.MCD*xx2 + p.const
yy_TLS = b1_TLS*xx1 + b2_TLS*xx2 + b0_TLS


# generate x-values for your regression line (two is sufficient)
xx1,xx2 = np.meshgrid(np.linspace(-0.15,0.15,20), np.linspace(-0.15,0.15,20))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_df["AAPL"], X_df["MCD"], y_df, s = 8, alpha = 0.5)
ax.plot_wireframe(xx1, xx2, yy_OLS, color = 'r', label = 'OLS')
ax.plot_wireframe(xx1, xx2, yy_TLS, color = 'b', label = 'TLS')

ax.set_xlim([-0.15,0.15])
ax.set_ylim([-0.15,0.15])
ax.set_zlim([-0.15,0.15])
ax.set_xlabel('AAPL')
ax.set_ylabel('MCD')
ax.set_zlabel('SP500')
ax.set_proj_type('ortho')
plt.legend(loc='lower right')

# plt.savefig('比较回归结果.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 正交回归，多元


import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


tickers = ['TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ', '^GSPC'];

# stock_levels_df = yf.download(tickers, start='2020-01-01', end='2020-12-31')

# stock_levels_df.to_csv('stock_levels_df.csv')
# stock_levels_df.to_pickle('stock_levels_df.pkl')

stock_levels_df = pd.read_pickle('stock_levels_df.pkl')
X_y_df = stock_levels_df['Adj Close'].pct_change()
X_y_df.dropna(inplace = True)

X_y_df.rename(columns={"^GSPC": "SP500"},inplace = True)
X_df = X_y_df.iloc[:,:-1]
y_df = X_y_df[['SP500']]



#%% TLS, matrix computation

SIMGA = X_y_df.cov()
Lambda, V = np.linalg.eig(SIMGA)
idx = Lambda.argsort()[::-1]
Lambda = Lambda[idx]
V = V[:,idx]

lambda_min = np.min(Lambda)
D = len(tickers[:-1])
b_TLS_ = -V[0:D,D]/V[D,D]
print(b_TLS_)

b0_TLS_ = y_df.mean().values - b_TLS_@X_df.mean().values
print(b0_TLS_)

b_TLS = np.hstack((b0_TLS_, b_TLS_))
labels = ['const'] + tickers[:-1]
b_df_TLS = pd.DataFrame(data=b_TLS.T, index=[labels], columns=['TLS'])


#%% OLS Regression
import statsmodels.api as sm

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())
b_df_OLS = model.fit().params
print(b_df_OLS)

b_df_OLS = pd.DataFrame(data=b_df_OLS.values, index=[labels], columns=['OLS'])


coeffs = pd.concat([b_df_TLS, b_df_OLS], axis=1, join="inner")
fig, ax = plt.subplots()
coeffs.plot.bar(ax = ax)
# h.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.axhline(y=0, color='r', linestyle='--')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 主元回归
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

# stock_levels_df = web.get_data_yahoo(tickers, start = '2020-07-01', end = '2020-12-31')
# stock_levels_df = yf.download(tickers, start='2020-01-01', end='2020-12-31')
stock_levels_df = pd.read_pickle('stock_levels_df.pkl')
stock_levels_df = stock_levels_df.rename(columns={"^GSPC": "SP500"})
stock_levels_df.round(2).head()

y_X_df = stock_levels_df['Adj Close'].pct_change()
y_X_df = y_X_df.dropna()


X_df = y_X_df[tickers[1:]]
y_df = y_X_df["SP500"]

labels = ['SP500','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];


#%% Lineplot of stock prices
# normalize the initial stock price levels to 1
normalized_stock_levels = stock_levels_df['Adj Close']/stock_levels_df['Adj Close'].iloc[0]

g = sns.relplot(data=normalized_stock_levels, dashes = False, kind="line") # , palette="coolwarm"
g.set_xlabels('Date')
g.set_ylabels('Normalized closing price')
g.set_xticklabels(rotation=45)


fig, ax = plt.subplots()
ax = sns.heatmap(y_X_df, cmap='RdBu_r', cbar_kws={"orientation": "vertical"}, yticklabels=False, vmin = -0.2, vmax = 0.2)
plt.title('[y, X]')


#%% distribution of column features of X
fig, axs = plt.subplots(2,2)

sns.kdeplot(ax = axs[0,0], data=y_X_df[labels[0:4]], fill=False, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
axs[0,0].set_xlim([-0.1,0.1])
axs[0,0].set_ylim([0, 45])

sns.kdeplot(ax = axs[0,1], data=y_X_df[labels[4:7]], fill=False, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
axs[0,1].set_xlim([-0.1,0.1])
axs[0,1].set_ylim([0, 45])

sns.kdeplot(ax = axs[1,0], data=y_X_df[labels[7:10]], fill=False, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
axs[1,0].set_xlim([-0.1,0.1])
axs[1,0].set_ylim([0, 45])

sns.kdeplot(ax = axs[1,1], data=y_X_df[labels[10:]], fill=False, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
axs[1,1].set_xlim([-0.1,0.1])
axs[1,1].set_ylim([0, 45])



#%% PCA

from sklearn.decomposition import PCA
pcamodel = PCA(n_components=4)
pca = pcamodel.fit_transform(X_df)

#%% Heatmap of V
V = pcamodel.components_.transpose()

fig, ax = plt.subplots()
ax = sns.heatmap(V, cmap='RdBu_r', xticklabels=['PC1','PC2','PC3','PC4'], yticklabels=list(X_df.columns), cbar_kws={"orientation": "vertical"}, vmin=-1, vmax=1, annot = True)
ax.set_aspect("equal")
plt.title('V')

fig, ax = plt.subplots()
ax = sns.heatmap(V.T@V, cmap='RdBu_r', xticklabels=['PC1','PC2','PC3','PC4'], yticklabels=['PC1','PC2','PC3','PC4'], cbar_kws={"orientation": "vertical"}, vmin=-1, vmax=1, annot = True)
ax.set_aspect("equal")
plt.title('V.T@V')

# Convert V array to dataframe
V_df = pd.DataFrame(data=V, columns = ['PC1','PC2','PC3','PC4'], index   = tickers[1:])

fig, ax = plt.subplots()
sns.lineplot(data=V_df, markers=True, dashes=False,palette = "husl")
plt.axhline(y=0, color='r', linestyle='-')

# V_df.to_excel('V.xlsx')

#%% projected data, Z
Z_df = X_df@V
Z_df = Z_df.rename(columns={0: "PC1", 1: "PC2", 2: "PC3", 3: "PC4"})

fig, ax = plt.subplots()
ax = sns.heatmap(Z_df, cmap='RdBu_r', cbar_kws={"orientation": "vertical"}, yticklabels=False, vmin = -0.2, vmax = 0.2)
plt.title('Z')

# distribution of column features of Z
fig, ax = plt.subplots()
sns.kdeplot(data=Z_df, fill=False, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
plt.title('Distribution of Z columns')

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Principal component')
ax1.set_ylabel('Variance explained (%)', color=color)
plt.plot(range(1,len(pcamodel.explained_variance_ratio_ )+1), np.cumsum(pcamodel.explained_variance_ratio_,), color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0,1])

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Variance', color=color)
plt.bar(range(1,len(pcamodel.explained_variance_ )+1), pcamodel.explained_variance_ )

ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
SIGMA_Z = Z_df.cov()

fig, axs = plt.subplots()
h = sns.heatmap(SIGMA_Z,cmap='RdBu_r', linewidths=.05, annot = True)
h.set_aspect("equal")

#%% approximate X
X_apx = Z_df@V.T

fig, ax = plt.subplots()
ax = sns.heatmap(X_apx, cmap='RdBu_r', cbar_kws={"orientation": "vertical"}, yticklabels=False, xticklabels = labels[1:], vmin = -0.2, vmax = 0.2)
plt.title('X_apx')

fig, ax = plt.subplots()
ax = sns.heatmap(X_df.to_numpy() - X_apx, cmap='RdBu_r', cbar_kws={"orientation": "vertical"}, yticklabels=False, xticklabels = labels[1:], vmin = -0.2, vmax = 0.2)
plt.title('Error')

#%% Least square regression
import statsmodels.api as sm

# add a column of ones
Z_plus_1_df = sm.add_constant(Z_df)

model = sm.OLS(y_df, Z_plus_1_df)
results = model.fit()
print(results.summary())

p_Z = model.fit().params
print(p_Z)

#%% coefficients

b_Z = p_Z[1:].T
b_X = V@b_Z # (12,)

b_X_df = pd.DataFrame(data = b_X.T, index = tickers[1:])

fig, ax = plt.subplots()
b_X_df.plot.bar()

b0 = y_df.mean() - X_df.mean().T@b_X


#%% increasing number of principal components

coeff_df = pd.DataFrame()
explained_array = []

num_PCs = [4,5,6,7,8,9]

for num_PC in num_PCs:
    pcamodel = PCA(n_components=num_PC)
    pca = pcamodel.fit_transform(X_df)
    V = pcamodel.components_.transpose()
    Z_df = X_df@V

    Z_plus_1_df = sm.add_constant(Z_df)
    model = sm.OLS(y_df, Z_plus_1_df)
    p_Z = model.fit().params

    b_Z = p_Z[1:].T
    b_X = V@b_Z
    b_X_df = pd.DataFrame(data = b_X.T, index = tickers[1:], columns = ['PC1~' + str(num_PC)])
    explained = np.sum(pcamodel.explained_variance_ratio_)
    print(explained)

    explained_array.append(explained)
    coeff_df = pd.concat([coeff_df, b_X_df], axis = 1)


fig, ax = plt.subplots()
h = sns.lineplot(data = coeff_df, markers = True, dashes = False, palette = "husl")
plt.axhline(y = 0, color = 'r', linestyle = '-')
h.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

fig, ax = plt.subplots()
h = sns.lineplot(data = coeff_df.T, markers = True, dashes = False, palette = "husl")
plt.axhline(y = 0, color = 'r', linestyle = '-')
h.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






































