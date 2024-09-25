

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 概率密度

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



x = np.arange(-10, 10+0.1, 0.1)

# Distance of random walk at t
mu = 0;
# W(0) = 0, which is the starting point of random walk

t = np.arange(1,21)

sigma = 1;
# standard normal distribution
sigma_series = np.sqrt(t)*sigma;

xx,tt = np.meshgrid(x,t);


fig, ax = plt.subplots()
plt.plot(t,sigma_series,marker = 'x')
plt.xlabel('t');
plt.ylabel('Sigma*sqrt(t)')
plt.xlim (t.min(),t.max())
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.xticks([1,5,10,15,20])


n_sigma = len(sigma_series);

colors = plt.cm.jet(np.linspace(0,1,n_sigma))
colors = np.flipud(colors)



fig, ax = plt.subplots()

pdf_matrix = [];

for i, t_i in zip(range(n_sigma),t):

    sigma = sigma_series[i];
    norm_ = norm(mu, sigma)
    pdf_x = norm_.pdf(x)

    plt.plot(x, pdf_x, color=colors[i], label='t = %s' % t_i)
    pdf_matrix.append(pdf_x)

plt.xlabel('x')
plt.ylabel('Probability density')
plt.xlim(-10,10)
plt.ylim(0,0.4)
plt.legend()
pdf_matrix = np.array(pdf_matrix)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 无漂移


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


N_steps = 200;
# number of steps
N_paths = 500;
# number of paths
sigma = 1
delta_t = 1

delta_X = np.random.normal(loc=0.0, scale=sigma*np.sqrt(delta_t), size=(N_steps,N_paths))
t_n = np.linspace(0,N_steps,N_steps+1,endpoint = True)*delta_t

X = np.cumsum(delta_X, axis = 0);
X_0 = np.zeros((1,N_paths))
X = np.vstack((X_0,X))

rows = 1
cols = 2



fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=(10,5), gridspec_kw={'width_ratios': [3, 1]})

ax1.plot(t_n, X, lw=0.25,color = '#0070C0')
ax1.axhline(y = 0, color = 'r')
ax1.plot(t_n, sigma*np.sqrt(t_n),color = 'r')
ax1.plot(t_n, -sigma*np.sqrt(t_n),color = 'r')
ax1.plot(t_n, 2*sigma*np.sqrt(t_n),color = 'r')
ax1.plot(t_n, -2*sigma*np.sqrt(t_n),color = 'r')
ax1.set_xlim([0,N_steps])
ax1.set_ylim([-60,60])
ax1.set_yticks([-50, 0, 50])
ax1.set_title('(a)', loc='left')
ax1.set_xlabel('t')

ax2 = sns.distplot(X[-1], rug=True, rug_kws={"color": "k", "alpha": 0.5, "height": 0.06, "lw": 0.5}, vertical=True, label='(b)', bins = 15)
# ax2 = sns.histplot(X[-1], rug=True, rug_kws={"color": "k", "alpha": 0.5, "height": 0.06, "lw": 0.5}, vertical=True, label='(b)', bins = 15)


ax2.set_yticks([-50, 0, 50])
ax2.set_title('(b)', loc='left')
ax2.set_ylim([-60,60])
#%% Snapshots at various time stamps


fig, axs = plt.subplots(1, 5, figsize=(20,4))
for i in np.linspace(0,4,5):
    i = int(i)
    X_i = X[int(i + 1)*40]
    E_X_i = X_i.mean()
    std_X_i = X_i.std()

    sns.distplot(X_i,rug=True, ax = axs[i],bins = 15,
                 hist_kws=dict(edgecolor="#0070C0", linewidth=0.25, facecolor = '#DBEEF3'),
                 rug_kws={"color": "k", "alpha": 0.5, "height": 0.06, "lw": 0.5})

    axs[i].plot(E_X_i, 0, 'xr')
    axs[i].axvline(x = E_X_i, color = 'r', ymax = 0.9)
    axs[i].axvline(x = E_X_i + std_X_i, color = 'r', ymax = 0.7)
    axs[i].axvline(x = E_X_i - std_X_i, color = 'r', ymax = 0.7)
    axs[i].axvline(x = E_X_i + 2*std_X_i, color = 'r', ymax = 0.5)
    axs[i].axvline(x = E_X_i - 2*std_X_i, color = 'r', ymax = 0.5)
    axs[i].set_xticks([-50, 0, 50])
    axs[i].set_xlim([-60,60])
    axs[i].set_ylim([0,0.08])










#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 平面随机行走


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.collections as mcoll

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline( x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc




N_steps = 1000;
# number of steps

delta_x = np.random.normal(loc=0.0, scale=1.0, size=(N_steps,1))
delta_y = np.random.normal(loc=0.0, scale=1.0, size=(N_steps,1))

disp_x = np.cumsum(delta_x, axis = 0);
disp_y = np.cumsum(delta_y, axis = 0);

disp_x = np.vstack(([0],disp_x))
disp_y = np.vstack(([0],disp_y))


fig, ax = plt.subplots()

# plt.plot(disp_x,disp_y);
lc = colorline(disp_x, disp_y, cmap='rainbow_r')
plt.plot(0,0,'kx', markersize = 20)
plt.ylabel('$x$');
plt.xlabel('$y$');
plt.axis('equal')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 有漂移



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N_steps = 200;
# number of steps
N_paths = 500;
# number of paths
sigma = 1
delta_t = 1
mu = 0.2



delta_X = np.random.normal(loc=0.0, scale=sigma*np.sqrt(delta_t), size=(N_steps,N_paths))
delta_X = delta_X + delta_t*mu
t_n = np.linspace(0,N_steps,N_steps+1,endpoint = True)*delta_t

X = np.cumsum(delta_X, axis = 0);
X_0 = np.zeros((1,N_paths))
X = np.vstack((X_0,X))



rows = 1
cols = 2

fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=(10,5), gridspec_kw={'width_ratios': [3, 1]})

ax1.plot(t_n, X, lw=0.25,color = '#0070C0')
ax1.plot(t_n, mu*t_n,color = 'r')
ax1.plot(t_n, sigma*np.sqrt(t_n) + mu*t_n,color = 'r')
ax1.plot(t_n, -sigma*np.sqrt(t_n) + mu*t_n,color = 'r')
ax1.plot(t_n, 2*sigma*np.sqrt(t_n) + mu*t_n,color = 'r')
ax1.plot(t_n, -2*sigma*np.sqrt(t_n) + mu*t_n,color = 'r')
ax1.set_xlim([0,N_steps])
ax1.set_ylim([-20,100])
ax1.set_yticks([-20, 0, 20, 40, 60, 80])
ax1.set_title('(a)', loc='left')
ax1.set_xlabel('t')

ax2 = sns.distplot(X[-1], rug=True, rug_kws={"color": "k",
                                                "alpha": 0.5,
                                                "height": 0.06,
                                                "lw": 0.5},
                   vertical=True, label='(b)', bins = 15)
ax2.set_yticks([-20, 0, 20, 40, 60, 80])
ax2.set_title('(b)', loc='left')
ax2.set_ylim([-20,100])


#%% Snapshots at various time stamps

fig, axs = plt.subplots(1, 5, figsize=(20,4))


for i in np.linspace(0,4,5):
    i = int(i)
    X_i = X[int(i + 1)*40]
    E_X_i = X_i.mean()
    std_X_i = X_i.std()

    sns.distplot(X_i,rug=True, ax = axs[i],bins = 15,
                 hist_kws=dict(edgecolor="b", linewidth=0.25),
                 rug_kws={"color": "k", "alpha": 0.5,
                          "height": 0.06, "lw": 0.5})

    axs[i].plot(E_X_i, 0, 'xr')
    axs[i].axvline(x = E_X_i, color = 'r', ymax = 0.9)
    axs[i].axvline(x = E_X_i + std_X_i, color = 'r', ymax = 0.7)
    axs[i].axvline(x = E_X_i - std_X_i, color = 'r', ymax = 0.7)
    axs[i].axvline(x = E_X_i + 2*std_X_i, color = 'r', ymax = 0.5)
    axs[i].axvline(x = E_X_i - 2*std_X_i, color = 'r', ymax = 0.5)
    axs[i].set_xticks([-20, 0, 20, 40, 60, 80])
    axs[i].set_xlim([-20,100])
    axs[i].set_ylim([0,0.07])



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 具有相关性的随机行走


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


N_steps = 200;
# number of steps

mu_1 = 0.2
mu_2 = 0.3

mu = np.matrix([mu_1, mu_2])

sigma_1 = 1
sigma_2 = 2
rho_1_2 = 0.8
delta_t = 1

SIGMA = np.matrix([[sigma_1**2, sigma_1*sigma_2*rho_1_2],
                   [sigma_1*sigma_2*rho_1_2, sigma_2**2]])

L = np.linalg.cholesky(SIGMA)
R = L.T

Z = np.random.normal(size = (N_steps,2))

delta_X = mu*delta_t + Z@R*np.sqrt(delta_t)

X = np.cumsum(delta_X, axis = 0);

X_0 = np.zeros((1,2))
X = np.vstack((X_0,X))

t_n = np.linspace(0,N_steps,N_steps+1,endpoint = True)*delta_t

rows = 1
cols = 2


fig, ax= plt.subplots()

ax.plot(t_n, X, lw=1)
ax.plot(t_n, mu_1*t_n,color = 'r', lw=0.25)
ax.plot(t_n, mu_2*t_n,color = 'r', lw=0.25)

ax.set_xlim([0,N_steps])
# ax.set_ylim([-20,100])
# ax.set_yticks([-20, 0, 20, 40, 60, 80])
ax.set_xlabel('t')
#%% scatter plot

import pandas as pd

delta_X_df = pd.DataFrame(data=delta_X, columns=["Delta x1", "Delta x2"])


fig, ax= plt.subplots()

# ax.scatter([delta_X[:,0]],[delta_X[:,1]])
sns.jointplot(data = delta_X_df, x = "Delta x1", y = "Delta x2")





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 几何布朗

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



N_steps = 200;
# number of steps
N_paths = 500;
# number of paths
sigma = 0.2
delta_t = 1
mu = 0.005
sigma = 0.05

X_0 = 10

t_n = np.linspace(0,N_steps,N_steps+1,endpoint = True)*delta_t

X = np.exp(
    (mu - sigma ** 2 / 2) * delta_t
    + sigma * np.random.normal(0, np.sqrt(delta_t), size=(N_steps,N_paths)))

X = np.vstack([np.ones((1,N_paths)), X])
X = X_0 * X.cumprod(axis=0)



rows = 1
cols = 2

fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=(10,5), gridspec_kw={'width_ratios': [3, 1]})

ax1.plot(t_n, X, lw=0.25,color = '#0070C0')
ax1.set_xlim([0,N_steps])
ax1.set_ylim([-25,200])
ax1.set_yticks([-20, 0, 50, 100, 150, 200])
ax1.set_title('(a)', loc='left')
ax1.set_xlabel('t')

ax2 = sns.distplot(X[-1], rug=True, rug_kws={"color": "k",
                                                "alpha": 0.5,
                                                "height": 0.06,
                                                "lw": 0.5},
                   vertical=True, label='(b)', bins = 20)
ax2.set_yticks([-20, 0, 50, 100, 150, 200])
ax2.set_title('(b)', loc='left')
ax2.set_ylim([-25,200])




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 股价模型

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader
import scipy.stats as stats


# Calibrate

df = pandas_datareader.data.DataReader(['sp500'], data_source='fred', start='08-01-2017', end='08-01-2021')
df = df.dropna()
df.to_csv('sp500.csv')
df.to_pickle('sp500.pkl')
print(df.tail())
#% Plot price levels of S&P 500


fig, ax = plt.subplots()

df['sp500'].plot()

plt.axhline(y=df['sp500'].max(), color= 'r', zorder=0)
plt.axhline(y=df['sp500'].min(), color= 'r', zorder=0)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

plt.xlabel('Date')
plt.ylabel('Adjusted closing price')
plt.show()


#%% daily log return

daily_log_r = df.apply(lambda x: np.log(x) - np.log(x.shift(1)))

daily_log_r = daily_log_r.dropna()

values = daily_log_r[1:]
mu_log_r, sigma_log_r = stats.norm.fit(values)




fig, ax = plt.subplots()

daily_log_r.plot(ax = ax)

plt.axhline(y=daily_log_r.max().values, color= 'r', zorder=0)
plt.axhline(y=0, color= 'r', zorder=0)
plt.axhline(y=daily_log_r.min().values, color= 'r', zorder=0)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

plt.xlabel('Date')
plt.ylabel('Daily return')
plt.show()

fig, ax = plt.subplots()

ax.hist(daily_log_r,bins=30, rwidth=0.85)
plt.axvline(x=0, color='k', linestyle='-')
plt.axvline(x=mu_log_r, color='r', linestyle='--')
plt.axvline(x=mu_log_r + sigma_log_r, color='r', linestyle='--')
plt.axvline(x=mu_log_r - sigma_log_r, color='r', linestyle='--')
plt.axvline(x=mu_log_r + 2*sigma_log_r, color='r', linestyle='--')
plt.axvline(x=mu_log_r - 2*sigma_log_r, color='r', linestyle='--')

plt.xlabel('Daily log return')
plt.ylabel('Frequency')
plt.title('$\mu_{daily}$ =  %1.3f \n $\sigma_{daily}$ = %1.3f' %(mu_log_r, sigma_log_r))



#%% Simulation

mu = mu_log_r*250
# annualized expected return

sigma = sigma_log_r*np.sqrt(250)
# square root of time rule

drift = (mu - sigma ** 2 / 2)
# drift

n = 50
# simulation steps

dt = 1/250
# assume 250 business days in a year

S0 = df['sp500'][-1]
# current stock price level

np.random.seed(1)

num_paths = 100;

wt = np.random.normal(0, np.sqrt(dt), size=(num_paths, n)).T

S = np.exp(drift*dt + sigma * wt)
S = np.vstack([np.ones(num_paths), S])
# Stack arrays in sequence vertically

S = S0 * S.cumprod(axis=0)

#%% plot paths and distribution for last day



import seaborn as sns

rows = 1
cols = 2
fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=(8,5), gridspec_kw={'width_ratios': [3, 1]})
ax1.plot(S)
ax1.set_yticks([3000,4000,5000,6000])
ax2 = sns.distplot(S[-1], rug=True, vertical=True)
ax2.set_yticks([3000,4000,5000,6000])




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 相关性股价模型


# initializations and download results
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yfinance as yf


tickers = ['TSLA','TSM','COST','NVDA','META','AMZN','AAPL','NFLX','GOOGL'];
# stock_levels_df = web.get_data_yahoo(tickers, start = '2020-08-01', end = '2021-08-01')

from datetime import datetime
startdate = datetime(2020,8,1)
enddate = datetime(2021,8,1)
stock_levels_df = yf.download(tickers = tickers,
                  start = startdate,
                  end = enddate)

stock_levels_df.to_csv("9_stocks_level.csv")


print(stock_levels_df.round(2).head())
print(stock_levels_df.round(2).tail())



fig = sns.relplot(data=stock_levels_df['Adj Close'],dashes = False,
            kind="line") # , palette="coolwarm"
fig.set_axis_labels('Date','Adjusted closing price')


# normalize the initial stock price levels to 1
normalized_stock_levels = stock_levels_df['Adj Close']/stock_levels_df['Adj Close'].iloc[0]

fig = sns.relplot(data=normalized_stock_levels,dashes = False,
            kind="line") # , palette="coolwarm"
fig.set_axis_labels('Date','Normalized closing price')

#%% daily log return

daily_log_r = stock_levels_df['Adj Close'].apply(lambda x: np.log(x) - np.log(x.shift(1)))

daily_log_r = daily_log_r.dropna()

#%% Variance-covariance matrix
# Compute the covariance matrix
cov_SIGMA = daily_log_r.cov()


# Set up the matplotlib figure
fig, ax = plt.subplots()

sns.heatmap(cov_SIGMA, cmap="coolwarm",
            square=True, linewidths=.05)
plt.title('Covariance matrix of historical data')

#%% correlation matrix

# Compute the correlation matrix
corr_P = daily_log_r.corr()
# Set up the matplotlib figure
fig, ax = plt.subplots()

sns.heatmap(corr_P, cmap="coolwarm",
            square=True, linewidths=.05, annot=True,
            vmax = 1,vmin = 0)
plt.title('Correlation matrix of historical data')


#%% Cholesky decomposition

import scipy.linalg

L = scipy.linalg.cholesky(cov_SIGMA, lower=True)
R = scipy.linalg.cholesky(cov_SIGMA, lower=False)

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(cov_SIGMA,cmap='coolwarm', cbar=False)
ax.set_aspect("equal")
plt.title('$\Sigma$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(L,cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('L')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(R,cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('R')


#%% Correlated stock price simulation

# convert daily parameters to yearly
mu_yearly = daily_log_r.mean()*250
R_yearly  = R*np.sqrt(250)
SIGMA_yearly = cov_SIGMA*250

n = 250
# simulation steps

dt = 1/250
# assume 250 business days in a year

S0 = stock_levels_df['Adj Close'].iloc[-1]

S0 = np.array(S0)
# current stock price levels

Z = np.random.normal(0, 1, size=(n, 9))
# only simulate one set of paths

drift = (mu_yearly - np.diag(SIGMA_yearly)/2)*dt;
drift = np.array(drift)

vol  = Z@R_yearly*np.sqrt(dt);

S = np.exp(drift + vol)

S = np.vstack([np.ones(9), S])
# add a layer of ones

S = S0 * S.cumprod(axis=0)
# compute the stock levels

Sim_df = pd.DataFrame(data=S, columns=tickers)
# convert the result to a dataframe

fig = sns.relplot(data=Sim_df,dashes = False,
            kind="line") # , palette="coolwarm"

plt.xlabel("$t$")
plt.ylabel("$S$")
plt.title('Simulated levels, one set of paths')

#%% Compute the correlation matrix of the simulated results

daily_log_sim = Sim_df.apply(lambda x: np.log(x) - np.log(x.shift(1)))
daily_log_sim = daily_log_sim.dropna()


# Compute the correlation matrix
corr_P_sim = daily_log_sim.corr()


# Set up the matplotlib figure
fig, ax = plt.subplots()

sns.heatmap(corr_P_sim, cmap="coolwarm",
            square=True, linewidths=.05, annot=True,
            vmax = 1,vmin = 0)

plt.title('Correlation matrix of simulated results')

# calculate the differences between historical and simulated
fig, ax = plt.subplots()

sns.heatmap(corr_P - corr_P_sim, cmap="coolwarm",
            square=True, linewidths=.05, annot=True,
            vmax = 1,vmin = 0)

plt.title('Differences, correlation matrix')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
















































































































































































































































































