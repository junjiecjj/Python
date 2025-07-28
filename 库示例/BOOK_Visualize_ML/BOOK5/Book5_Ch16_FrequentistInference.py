



#%% Bk5_Ch16_01
# 图 8. 每次抛 n = 20 枚色子, 中心极限定理：渐近于正态分布
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mean_array = []
num_dices  = 20 # n = 5, 10, 20
num_trials = 100000

# each trial: 10 dices and calculate mean
for i in np.arange(num_trials):
    sample_i = np.random.randint(low = 1, high = 6 + 1, size=(num_dices))
    mean_i   = sample_i.mean()
    mean_array.append(mean_i)

# plot the histogram of mean values at 50, 500, 5000 trials
for j in [1000, 10000, 100000]: # m
    mean_array_j = mean_array[0:j]
    fig, ax = plt.subplots()
    sns.histplot(mean_array_j, kde = True, stat="density", binrange = [1, 6], bins = 100, binwidth = 0.1)
    mean_array_j = np.array(mean_array_j)
    mu_mean_array_j = mean_array_j.mean()
    ax.axvline(x = mu_mean_array_j, color = 'r',linestyle = '--')
    sigma_mean_array_j = mean_array_j.std()
    ax.axvline(x = mu_mean_array_j + sigma_mean_array_j, color = 'r',linestyle = '--')
    ax.axvline(x = mu_mean_array_j - sigma_mean_array_j, color = 'r',linestyle = '--')

    plt.xlim(1, 6)
    plt.ylim(0, 1)
    plt.grid()

#%% Bk5_Ch16_02
# 图 9. 随机数分布
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# create a population
num_population = 100000
X1 = np.random.normal(loc = -5, scale = 1.0, size = int(num_population/2)) # (50000,)
X2 = np.random.normal(loc = 5,  scale = 3,   size = int(num_population/2)) # (50000,)

X = np.concatenate((X1, X2), axis = None) # (100000,)

fig, ax = plt.subplots()
sns.kdeplot(X, fill = True)
mu_X = X.mean()
ax.axvline(x = mu_X, color = 'r',linestyle = '--')
sigma_X = X.std()
ax.axvline(x = mu_X + sigma_X, color = 'r',linestyle = '--')
ax.axvline(x = mu_X - sigma_X, color = 'r',linestyle = '--')

plt.grid()

#%% 图 10. 每次抽取 10 个样本
num_draws  = 10
num_trials = 5000

mean_array = []
# each trial: 10 dices and calculate mean
for i in np.arange(num_trials):
    indice_i = np.random.randint(low = 0, high = num_population, size=(num_draws))
    sample_i = X[indice_i]

    mean_i   = sample_i.mean()
    mean_array.append(mean_i)

# plot the histogram of mean values at 50, 500, 5000 trials
for j in [50, 500, 5000]:
    mean_array_j = mean_array[0:j]
    fig, ax = plt.subplots()
    sns.histplot(mean_array_j, kde = True, stat="density", binrange = [-10,10], binwidth = 0.2)
    mean_array_j = np.array(mean_array_j)
    mu_mean_array_j = mean_array_j.mean()

    ax.axvline(x = mu_mean_array_j, color = 'r',linestyle = '--')
    sigma_mean_array_j = mean_array_j.std()
    ax.axvline(x = mu_mean_array_j + sigma_mean_array_j, color = 'r',linestyle = '--')
    ax.axvline(x = mu_mean_array_j - sigma_mean_array_j, color = 'r',linestyle = '--')

    plt.xlim(-10, 10)
    plt.ylim(0, 0.3)
    plt.grid()


#%% distributions of mean of mean
# 图 11. 随着试验次数增大，均值分布逐渐趋向正态
num_trials = 5000
fig, ax = plt.subplots()

for num_draws in [4, 8, 16]: # 每次抽取 n 个样本
    mean_array = []
    for i in np.arange(num_trials):
        indice_i = np.random.randint(low = 0, high = num_population, size=(num_draws))
        sample_i = X[indice_i]

        mean_i   = sample_i.mean()
        mean_array.append(mean_i)
        # finishing the generation of mean array
    sns.kdeplot(mean_array, fill = True)

plt.xlim(-10,10)
plt.ylim(0,0.3)
plt.grid()

#%% SE: standard error
# 图 12. 标准误随 n 变化
num_trials = 5000
SE_array = []
n_array = np.linspace(4, 100, 25)
for num_draws in n_array:
    mean_array = []
    for i in np.arange(num_trials):
        indice_i = np.random.randint(low = 0,  high = num_population, size=(int(num_draws)))
        sample_i = X[indice_i]
        mean_i   = sample_i.mean()
        mean_array.append(mean_i)
        # finishing the generation of mean array
    mean_array = np.array(mean_array)
    SE_i = mean_array.std()
    SE_array.append(SE_i)

fig, ax = plt.subplots()
ax.plot(n_array,SE_array, marker = 'x', markersize = 12)
ax.set_xlim(4,100)
ax.set_ylim(0,3)
plt.grid()
plt.show()
plt.close()


#%% Bk5_Ch16_03
# 图 15. lnL(θ1, θ2) 曲面和最大值点位置
import numpy as np
from sympy import symbols, ln, simplify, lambdify, diff, solve, Float
import matplotlib.pyplot as plt

theta_1, theta_2 = symbols('theta_1 theta_2')
samples = [-2.5, -5, 1, 3.5, -4, 1.5, 5.5]
mu = np.mean(samples)

print(mu)
n = len(samples)
bias_std = np.std(samples)
bias_var = bias_std**2
print(bias_var)

A = 0
for i in np.arange(n):
    term_i = (samples[i] - theta_1)**2
    A = A + term_i

A = simplify(A)
print(A)
lnL = -n/2*np.log(2*np.pi) - n/2*ln(theta_2) - A/2/theta_2

####
lnL = simplify(lnL)
print(lnL)

theta_1_array = np.linspace(mu-3, mu+3, 40)
theta_2_array = np.linspace(bias_var*0.8, bias_var*1.2, 40)

theta_11,theta_22 = np.meshgrid(theta_1_array, theta_2_array)
lnL_fcn = lambdify((theta_1, theta_2), lnL)
lnL_matrix = lnL_fcn(theta_11, theta_22)

####
# first-order partial differential
df_dtheta_1 = diff(lnL, theta_1)
print(df_dtheta_1)

df_dtheta_2 = diff(lnL, theta_2)
print(df_dtheta_2)

# solution of (theta_1,theta_2)
sol = solve([df_dtheta_1, df_dtheta_2], [theta_1, theta_2])
print(sol)

theta_1_star = sol[0][0]
theta_2_star = sol[0][1]

theta_1_star = theta_1_star.evalf()
theta_2_star = str(theta_2_star)
theta_2_star = eval(theta_2_star)

print(theta_1_star)
print(theta_2_star)

lnL_min = lnL_fcn(theta_1_star,theta_2_star)
print(lnL_min)
#####
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(theta_11, theta_22, lnL_matrix, color = [0.5,0.5,0.5], linewidth = 0.25)
plt.plot(theta_1_star, theta_2_star, lnL_min, marker = 'x', markersize = 12)
colorbar = ax.contour(theta_11, theta_22, lnL_matrix, 30, cmap = 'RdYlBu_r')

fig.colorbar(colorbar, ax=ax)
ax.set_proj_type('ortho')
ax.set_xlabel('$\\theta_1$, $\\mu$')
ax.set_ylabel('$\\theta_2$, $\\sigma^2$')

plt.tight_layout()
ax.set_xlim(theta_11.min(), theta_11.max())
ax.set_ylim(theta_22.min(), theta_22.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.show()

fig, ax = plt.subplots()
colorbar = ax.contourf(theta_11, theta_22, lnL_matrix, 30, cmap='RdYlBu_r')
fig.colorbar(colorbar, ax=ax)
plt.plot(theta_1_star, theta_2_star, marker = 'x', markersize = 12)

ax.set_xlim(theta_11.min(), theta_11.max())
ax.set_ylim(theta_22.min(), theta_22.max())
ax.set_xlabel('$\\theta_1$, $\\mu$')
ax.set_ylabel('$\\theta_2$, $\\sigma^2$')
# plt.gca().set_aspect('equal', adjustable='box')
plt.show()
plt.close()


#%% Bk5_Ch16_04
# 16.6 区间估计：总体方差已知，均值估计
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x   = np.linspace(start = -4, stop = 4, num = 200)
f_x = stats.norm.pdf(x) # PDF of standard normal distribution

alpha = 0.05

### population standard deviation is known, or large sample size
### Get the critical values, two-tailed
crit_value = stats.norm.ppf(q = 1 - alpha/2)

fig, ax = plt.subplots()

ax.plot(x, f_x, color = "#0070C0")
ax.fill_between(x[np.logical_and(x >= -crit_value, x <= crit_value)], f_x[np.logical_and(x >= -crit_value, x <= crit_value)], color = "#DBEEF3")

ax.axvline(x = crit_value,  color = 'r', linestyle = '--')
ax.plot(crit_value, 0, marker = 'x', color = 'k', markersize = 12)
ax.axvline(x = -crit_value, color = 'r', linestyle = '--')
ax.plot(-crit_value, 0, marker = 'x', color = 'k', markersize = 12)

ax.fill_between(x[x <= -crit_value], f_x[x <= -crit_value], color = "#FF9980")
ax.fill_between(x[x >= crit_value],  f_x[x >= crit_value], color = "#FF9980")

ax.set_title("Population sigma known, $\\alpha = 0.05$, two-tailed")
ax.set_xlim(-4,4)
ax.set_ylim(0,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
plt.close()

### left-sided
crit_value = stats.norm.ppf(q = 1 - alpha)
fig, ax = plt.subplots()
ax.plot(x, f_x, color = "#0070C0")
ax.fill_between(x[x >= -crit_value], f_x[x >= -crit_value], color = "#DBEEF3")
ax.axvline(x = -crit_value, color = 'r', linestyle = '--')
ax.plot(-crit_value,0,marker = 'x', color = 'k', markersize = 12)

ax.fill_between(x[x <= -crit_value], f_x[x <= -crit_value], color = "#FF9980")
ax.set_title("Population sigma known, $\\alpha = 0.05$, left-tailed")
ax.set_xlim(-4,4)
ax.set_ylim(0,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
plt.close()

### right-sided
crit_value = stats.norm.ppf(q = 1 - alpha)

fig, ax = plt.subplots()
ax.plot(x, f_x, color = "#0070C0")
ax.fill_between(x[x <= crit_value], f_x[x <= crit_value], color = "#DBEEF3")
ax.axvline(x = crit_value, color = 'r', linestyle = '--')
ax.plot(crit_value,0,marker = 'x', color = 'k', markersize = 12)
ax.fill_between(x[x >= crit_value], f_x[x >= crit_value],  color = "#FF9980")
ax.set_title("Population sigma known, $\\alpha = 0.05$, right-tailed")
ax.set_xlim(-4,4)
ax.set_ylim(0,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
plt.close()


#%% Bk5_Ch16_05
# 16.7 区间估计：总体方差未知，均值估计
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x   = np.linspace(start = -4, stop = 4, num = 200)
# In practice, population standard deviation is rarely known
n = 6
f_x_t = stats.t.pdf(x, df = n-1)
# PDF of student t distribution
f_x_norm = stats.norm.pdf(x)

alpha = 0.05

## Get the critical values, two-tailed
crit_value = stats.t.ppf(q = 1-alpha/2, df = n-1)

fig, ax = plt.subplots()

plt.plot(x, f_x_t, color = "#0070C0")
plt.plot(x, f_x_norm, color = "k", linestyle = '--')

plt.fill_between(x[np.logical_and(x >= -crit_value, x <= crit_value)], f_x_t[np.logical_and(x >= -crit_value, x <= crit_value)], color = "#DBEEF3")

ax.axvline(x = crit_value,  color = 'r', linestyle = '--')
plt.plot(crit_value, 0,marker = 'x', color = 'k', markersize = 12)
ax.axvline(x = -crit_value, color = 'r', linestyle = '--')
plt.plot(-crit_value,0,marker = 'x', color = 'k', markersize = 12)

plt.fill_between(x[x <= -crit_value], f_x_t[x <= -crit_value], color = "#FF9980")
plt.fill_between(x[x >= crit_value],  f_x_t[x >= crit_value], color = "#FF9980")

plt.title("Population sigma unknown, $\\alpha = 0.05$, two-tailed")

ax.set_xlim(-4,4)
ax.set_ylim(0,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

#%% Bk5_Ch16_06
# 16.8 区间估计：总体均值未知，方差估计
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x   = np.linspace(start = 0, stop = 16, num = 1000)
# In practice, population standard deviation is rarely known
n = 6
f_x_chi2 = stats.chi2.pdf(x, df = n-1)

alpha = 0.05

#%% Get the critical values, two-tailed
crit_value_right = stats.chi2.ppf(q = 1-alpha/2, df = n-1)
crit_value_left  = stats.chi2.ppf(q = alpha/2, df = n-1)

fig, ax = plt.subplots()

plt.plot(x, f_x_chi2, color = "#0070C0")
plt.fill_between(x[np.logical_and(x >= crit_value_left, x <= crit_value_right)], f_x_chi2[np.logical_and(x >= crit_value_left, x <= crit_value_right)], color = "#DBEEF3")

ax.axvline(x = crit_value_right,  color = 'r', linestyle = '--')
plt.plot(crit_value_right, 0,marker = 'x', color = 'k', markersize = 12)
ax.axvline(x = crit_value_left, color = 'r', linestyle = '--')
plt.plot(crit_value_left,0,marker = 'x', color = 'k', markersize = 12)

plt.fill_between(x[np.logical_and(x >= 0, x <= crit_value_left)], f_x_chi2[np.logical_and(x >= 0, x <= crit_value_left)], color = "#FF9980")

plt.fill_between(x[np.logical_and(x <= x.max(), x >= crit_value_right)], f_x_chi2[np.logical_and(x <= x.max(), x >= crit_value_right)], color = "#FF9980")

ax.set_xlim(0,16)
ax.set_ylim(0,0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)





































































































































































































































































