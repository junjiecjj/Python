

#%% Bk5_Ch22_01
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def target_PDF(likelihood, prior, n, s, theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        likelihood_fcn = likelihood(n, theta).pmf(s)
        prior_fcn = prior.pdf(theta)
        posterior_fcn = likelihood_fcn * prior_fcn
        return posterior_fcn

def data_generator(num_iterations, n, s, theta_0, likelihood, prior, drop_unstable = True, sigma = 0.3):
    ### container for sample data
    samples = [theta_0]
    num_accepted = 0
    ### Metropolis-Hastings 采样, Book05_chap22_page5
    theta_now  = theta_0
    for idx in range(num_iterations):
        ## 它通过在取值空间取任意值作为起始点，按照先验分布计算概率密度，计算起始点的概率密度。然后随机移动到下一点时，计算当前点的概率密度。移动的步伐一般从正态分布中抽取。
        delta_theta = stats.norm(0, sigma).rvs()
        theta_next  = theta_now + delta_theta
        numerator   = target_PDF(likelihood, prior, n, s, theta_next)
        denominator = target_PDF(likelihood, prior, n, s, theta_now)
        ## 接着，计算当前点和起始点概率密度的比值 ρ，并产生 (0,1) 之间服从连续均匀的随机数 u。最后，对比 ρ 与产生的随机数 u 的大小来判断是否保留当前点。当前者大于后者，接受当前点，反之则拒绝当前点。
        ## 这个过程一直循环，直到获得能被接受后验分布。这一步和本书第 15 章介绍的“接受-拒绝抽样”本质上一致。
        rho = min(1, numerator/denominator)
        u_idx = np.random.uniform()
        if u_idx < rho:
            num_accepted += 1
            theta_now = theta_next
        samples.append(theta_now)
    if drop_unstable:
        nmcmc = len(samples)//2
        samples = samples[nmcmc:]
    return samples

### initialization
theta_array = np.linspace(0, 1, 500)
# number of animal samples:
n = 200
# number of rabbits in data:
s = 60
# initiall guess
theta_0 = 0.1
# number of iterations
num_iterations = 5000
# prior distribution assumption: 1:1 ratio, i.e., alpha = beta in Beta(alpha, beta) distribution
alpha_arrays = [1, 2, 8, 16, 32, 64]

for alpha in alpha_arrays:
    beta  = alpha
    # distributions
    likelihood = stats.binom
    prior = stats.beta(alpha, beta)
    post = stats.beta(s+alpha, n-s+beta)
    # analytical to compare with

    # generate random data
    samples = data_generator(num_iterations, n, s, theta_0, likelihood, prior)

    # compare analytical vs samples
    fig, ax = plt.subplots(figsize = (12,6))
    ax.hist(samples, 20, histtype = 'step', density = True, linewidth = 2, color = 'b', label = 'Posterior')
    ax.plot(theta_array, post.pdf(theta_array), c = 'b', linestyle = '--', alpha = 0.25, label = 'Analytical posterior')
    ax.fill_between(theta_array, 0, post.pdf(theta_array), color = 'b', alpha = 0.2)

    ax.hist(prior.rvs(len(samples)), 20, histtype = 'step', density = True, linewidth = 2, color = 'r', label = 'Prior');
    ax.plot(theta_array, prior.pdf(theta_array), c = 'r', linestyle = '--', alpha = 0.25, label = 'Analytical prior')
    ax.fill_between(theta_array, 0, prior.pdf(theta_array), color='r', alpha = 0.2)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 15])
    ax.set_yticks([0,5,10,15])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Density')
    ax.set_title('Alpha = %0.0f' %alpha)
    plt.legend(loc='best')
    plt.show()
plt.close('all')

### stability distributions
alpha = 16
beta  = alpha
likelihood = stats.binom
prior = stats.beta(alpha, beta)

num_iterations = 200
samples_5_sets = [data_generator(num_iterations, n, s, theta_0, likelihood, prior, drop_unstable = False, sigma = 0.1) for theta_0 in np.linspace(0.1, 0.9, 5)]

# Convergence of multiple MCMC chains
fig, ax = plt.subplots(figsize = (12,6))
for data_idx in samples_5_sets:
    ax.plot(data_idx, '-o')

ax.set_xlim([0, num_iterations])
ax.set_ylim([0, 1]);
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\theta$')
plt.show()
plt.close()

#%% Bk5_Ch22_02
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm

alphas = np.array([2, 2])
# Dir(2,2), or Beta(2, 2)

data = np.array([2, 3])     # n = 5
# data = np.array([90,110]) # = 200
animals = ['Rabbit', 'Chicken']

### Create model object
with pm.Model() as model:
    # prior: Dirichlet distribution
    prior = pm.Dirichlet('parameters', a = alphas, shape = 2)
    # likelihood: multinomial distribution
    observed_data = pm.Multinomial('observed_data', n = data.sum(), p = prior, shape = 2, observed = data)
### Simulate posterior distribution
with model:
    # 1000 sample data from 2 chains
    # First 200 samples are discarded
    trace = pm.sample(draws = 1000, chains = 2, tune = 200, discard_tuned_samples = True)

### Print results
summary = pm.summary(trace)
summary.index = animals

### Posterior distributions
trace_df = pd.DataFrame(trace['parameters'], columns = animals)
# trace plot
ax = pm.traceplot(trace, figsize = (16, 8), combined = True)
# Flag for combining multiple chains into a single chain.
# If False (default), chains will be plotted separately
ax[0][0].set_xlabel(r'$\theta$')
ax[0][0].set_ylabel('PDF')
ax[0][0].set_xlim(0,1)
ax[0][1].set_xlabel('Iteration')
# plt.savefig('Rabbit_Chicken_Posterior_Trace_plot_5.svg')

# Posterior hist
ax = pm.plot_posterior(trace, kind="hist", figsize = (18, 6), edgecolor = 'k')
for i, a in enumerate(animals):
    ax[i].set_title(a)
    ax[i].set_xlim(0,1)
# plt.savefig('Rabbit_Chicken_Posterior_Hist_5.svg')

# Posterior KDE
ax = pm.plot_posterior(trace, kind="kde", figsize = (18, 6))
for i, a in enumerate(animals):
    ax[i].set_title(a)
    ax[i].set_xlim(0,1)
# plt.savefig('Rabbit_Chicken_Posterior_KDE_5.svg')

#%% Bk5_Ch22_03
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm

### Prior
alphas = np.array([2,2,2])
# Dir(2, 2, 2)

## Data
data = np.array([3,6,1])    # n = 10
data = np.array([65,115,20]) # = 200
animals = ['Rabbit', 'Chicken', 'Piglet']

### Create model object
with pm.Model() as model:
    # prior: Dirichlet distribution
    prior = pm.Dirichlet('parameters', a=alphas, shape = 3)
    # likelihood: multinomial distribution
    observed_data = pm.Multinomial('observed_data', n = data.sum(), p = prior, shape = 3, observed = data)
### Simulate posterior distribution
with model:
    # 1000 sample data from 2 chains
    # First 200 samples are discarded
    trace = pm.sample(draws=1000, chains=2, tune=200, discard_tuned_samples=True)

### Print results
summary = pm.summary(trace)
summary.index = animals


### Posterior distributions
trace_df = pd.DataFrame(trace['parameters'], columns = animals)

# trace plot
ax = pm.traceplot(trace, figsize = (16, 8), combined = True);
# Flag for combining multiple chains into a single chain.
# If False (default), chains will be plotted separately

ax[0][0].set_xlabel(r'$\theta$')
ax[0][0].set_ylabel('PDF');
ax[0][0].set_xlim(0,1);
ax[0][1].set_xlabel('Iteration');

# plt.savefig('Rabbit_Chicken_Piglet_Posterior_Trace_plot_200.svg')

# Posterior hist
ax = pm.plot_posterior(trace, kind="hist", figsize = (18, 6), edgecolor = 'k');
for i, a in enumerate(animals):
    ax[i].set_title(a)
    ax[i].set_xlim(0,1)
# plt.savefig('Rabbit_Chicken_Piglet_Posterior_Hist_200.svg')

# Posterior KDE
ax = pm.plot_posterior(trace, kind="kde", figsize = (18, 6));
for i, a in enumerate(animals):
    ax[i].set_title(a)
    ax[i].set_xlim(0,1)
# plt.savefig('Rabbit_Chicken_Piglet_Posterior_KDE_200.svg')





















































































































































































































































