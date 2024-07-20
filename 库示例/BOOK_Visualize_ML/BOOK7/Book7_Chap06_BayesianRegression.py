



# 贝叶斯线性回归
# ! pip install pymc3

import pymc  as pm
# import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RANDOM_SEED = 1
rng = np.random.default_rng(RANDOM_SEED)

size = 50
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x

# add noise and generate observations (sample data)
y = true_regression_line + rng.normal(scale=0.5, size=size)

data = pd.DataFrame(dict(x=x, y=y))


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, xlabel="x", ylabel="y")
plt.plot(x, y, 'b.');
ax.plot(x, true_regression_line, color = 'k', label="True regression line", lw=2.0)
plt.legend(loc=0)
plt.xlim(0,1)
plt.xlabel('$x$')
plt.ylabel('$y$', rotation=0)
# plt.savefig('Bayesian regression scatter with true regression.svg')

# use pymc3 and create a Bayesian model
basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    intercept = pm.Normal('alpha', mu=0, sigma=20) # b_0
    slope     = pm.Normal('beta', mu=0, sigma=20) # b_1
    sigma     = pm.HalfNormal('sigma', sigma=20) # or pm.HalfCauchy
    mu = intercept + slope*x
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)


# understand half=normal distribution
import scipy.stats as st
from matplotlib import cm # Colormaps

fig = plt.figure(figsize=(5, 5))
x_ = np.linspace(0, 5, 200)
sigma_array = np.linspace(0.2,2,10)
colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(sigma_array)))

for i in range(0,len(sigma_array)):
    sigma_i = sigma_array[i]
    pdf = st.halfnorm.pdf(x_, scale=sigma_i)
    plt.plot(x_, pdf, color = colors[int(i)], label = r'$\sigma$ = %.1f' %sigma_i)

plt.xlabel('x')
plt.ylabel('PDF')
plt.legend(loc=1)
plt.xlim(0,5)
plt.ylim(0,5)
# plt.savefig('Bayesian regression half-normal.svg')


# simulate
with basic_model:
    # draw 2000 posterior samples from 2 chains
    trace = pm.sample(draws=1000, chains=2, tune=200, discard_tuned_samples=True)

# plot scatter and contour with marginals

sns.jointplot(x = trace['alpha'], y = trace['beta'], kind="kde", cmap="Blues", shade=True)
# sns.jointplot(data = trace,
#                   x = 'alpha', y = 'beta', ax = ax)
# g.plot_joint(sns.kdeplot, color="r", cmap="Blues", shade=True, ax = ax)

plt.xlabel("beta[0]")
plt.ylabel("beta[1]")
# plt.savefig('Bayesian regression jointplot.svg')



# Posterior hist
ax = pm.plot_posterior(trace, kind="hist", figsize = (18, 6), edgecolor = 'k');

# Posterior KDE
ax = pm.plot_posterior(trace, kind="kde", figsize = (18, 6));
# plt.savefig('Bayesian regression KDE.svg')



true_regression_line

fig, ax = plt.subplots(figsize=(5, 5))
idx_array = range(0, len(trace['alpha']), 40)
alpha_m = trace['alpha'].mean()
beta_m = trace['beta'].mean()

for idx in idx_array:
    plt.plot(x, trace['alpha'][idx] + trace['beta'][idx] *x, c='k', alpha = 0.1);

ax.plot(x, true_regression_line,
    color = 'k',
    label="True regression line", lw=2.0)

label_2 = 'Prediction: y = {:.2f} + {:.2f}* x'.format(alpha_m, beta_m)
plt.plot(x, alpha_m + beta_m * x, c='r', label=label_2)
plt.xlabel('$x$')
plt.ylabel('$y$', rotation=0)
plt.legend(loc=2)
plt.xlim(0,1)



























































































































































































































































































































































