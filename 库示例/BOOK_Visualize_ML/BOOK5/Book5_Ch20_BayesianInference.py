



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk5_Ch20_01
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

alpha = 1 # degree of belief # 1, 2, 16
true_percentage = 0.45 # 0.3
K = 2000
Data_all_trials = stats.bernoulli.rvs(true_percentage, size = K)

#%% visualize data of trials
# 图 8. 某次试验的模拟结果，先验分布为 Beta(1, 1)
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 30))

trials_array = np.arange(0, K) + 1
mask = (Data_all_trials == 1)

axs[0].plot(Data_all_trials[mask], trials_array[mask], color = 'b', linestyle = None, marker = '.', markersize = 20, label = 'Rabbit')
axs[0].plot(Data_all_trials[~mask], trials_array[~mask], color = 'r', linestyle = None, marker = '.', markersize = 20, label = 'Chicken')
axs[0].plot(Data_all_trials, trials_array, color = [0.8, 0.8, 0.8])
axs[0].set_ylim(1, K)
axs[0].set_xlim(-0.5, 1.5)
axs[0].set_xticks([0, 1])

axs[0].set_ylabel("Number of trials", rotation = 90)
# axs[0].yaxis.tick_right()
# axs[0].yaxis.set_label_position("right")
axs[0].set_xlabel("Result of each trial")
# axs[0].invert_xaxis()
axs[0].legend()
# plt.setp(axs[0].get_xticklabels(), rotation=90, va="top", ha="center")
# plt.setp(axs[0].get_yticklabels(), rotation=90, va="center", ha="left")
ratio_rabbits = np.cumsum(Data_all_trials)/trials_array
ratio_chickens = 1 - ratio_rabbits

axs[1].plot(ratio_rabbits, trials_array, color = 'b', label = 'Rabbit')
axs[1].plot(ratio_chickens, trials_array, color = 'r', label = 'Chicken')
axs[1].set_ylim(1, K)
axs[1].set_xlim(0, 1)
axs[1].set_xticks([0, 0.5, 1])
axs[1].set_ylabel("Number of trials", rotation = 90)
# axs[1].yaxis.tick_right()
# axs[1].yaxis.set_label_position("right")
axs[1].set_xlabel("Ratio")
# axs[1].invert_xaxis()
axs[1].legend()

# plt.setp(axs[1].get_xticklabels(), rotation=90, va="top", ha="center")
# plt.setp(axs[1].get_yticklabels(), rotation=90, va="center", ha="left")
plt.show()
plt.close()

#%% Continuous variations of the posterior ridgeline style
# 图 8. 某次试验的模拟结果，先验分布为 Beta(1, 1)
from matplotlib.pyplot import cm
theta_array = np.linspace(0, 1, 500)

num_animals_array = np.arange(0, K + 5, 50)
num_animals_array = num_animals_array[::-1]
# reverse the sequence of layers
colors = cm.rainbow_r(np.linspace(0, 1, len(num_animals_array)))

fig, ax = plt.subplots(figsize=(8, 60))
for idx, num_animals_idx in enumerate(num_animals_array):
    height = num_animals_idx
    # random data generator
    data_idx = Data_all_trials[0:num_animals_idx]
    # actual percentage of rabbits is 30%

    num_rabbits_idx = data_idx.sum() # s
    posterior_pdf = stats.beta.pdf(theta_array,
                      num_rabbits_idx + alpha,  # s + alpha
                      num_animals_idx - num_rabbits_idx + alpha) # n - s + alpha
    ratio = 1.2
    ax.plot(theta_array, posterior_pdf * ratio + height, color = [0.6,0.6,0.6])
    ax.fill_between(theta_array, height, posterior_pdf * ratio + height, color=colors[idx])

ax.set_xlim(0,1)
ax.set_xlabel('Posterior')
ax.set_ylabel('Number of trials')
plt.show()
plt.close()

#%% snapshots of posterior curves locations of snapshots
# 图 9. 九张不同节点的后验概率分布曲线快照，先验分布为 Beta(1, 1)
num_animals_array = [0, 1, 2, 3, 4, 5, 10, 100, 200]
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
for ax_idx, num_animals_idx in zip(axs.ravel(), num_animals_array):
    # random data generator
    data_idx = Data_all_trials[0:num_animals_idx]
    # actual percentage of rabbits is 30%
    num_rabbits_idx = data_idx.sum() # s
    posterior_pdf = stats.beta.pdf(theta_array,
                      num_rabbits_idx + alpha,  # s + alpha
                      num_animals_idx - num_rabbits_idx + alpha) # n - s + alpha
    loc_max = theta_array[np.argmax(posterior_pdf)]
    # location of MAP

    ax_idx.plot(theta_array, posterior_pdf)
    ax_idx.axvline(x = loc_max, color = 'r', linestyle = '--')
    ax_idx.set_title("Number of animals: %d; number of rabbits: %d" % (num_animals_idx, num_rabbits_idx))

    ax_idx.set_xlabel('Percentage of rabbits, $\u03B8$')
    ax_idx.fill_between(theta_array, 0, posterior_pdf, color="#DEEAF6")
    ax_idx.axvline(x = true_percentage, color = 'k', linestyle = '--')
    ax_idx.set_xlim(0,1)
    ax_idx.set_yticks([0,5,10,15])
    ax_idx.set_ylim(0,15)
plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 图 15. 五个不同参数 α 取不同值时 Beta(α, α) 分布 PDF 图像
theta_array = np.linspace(0, 1, 100)
alpha_list = [0.5, 1, 2, 4, 8, 16]
fig, axs = plt.subplots(nrows=1, ncols=len(alpha_list), figsize=(len(alpha_list)*4, 3))
for ax_idx, alpha in zip(axs.ravel(), alpha_list):
    # random data generator
    # data_idx = Data_all_trials[0:num_animals_idx]
    # actual percentage of rabbits is 30%
    # num_rabbits_idx = data_idx.sum() # s
    pdf = stats.beta.pdf(theta_array,
                       alpha,
                       alpha)
    ax_idx.plot(theta_array, pdf)
    ax_idx.set_title(f"alpha = {alpha}")

plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 图 17. 似然分布，n = 20
import scipy.stats as stats


theta_array = np.linspace(0, 1, 100)
n = 20
slist = np.arange(1, 20, 2)
colors = cm.rainbow(np.linspace(0, 1, len(slist)))

fig, ax = plt.subplots(1, 2, figsize = (12, 6))
for i, s in enumerate(slist):
    ax[0].plot(theta_array, stats.binom.pmf(s, n, theta_array), label=f"{s}", c=colors[i])  # Book06_chap20_(29)
    # ax[1].plot(theta_array, theta_array**s * (1-theta_array)**(n-s), label=f"{s}", c=colors[i])  # Book06_chap20_(29)
plt.legend(ncols = 2)
plt.show()
plt.close()

# 图 18. 似然分布和 MLE 优化解的位置，n = 20
theta_array = np.linspace(0, 1, 100)
n = 20
slist = np.arange(0, 21, 1)
colors = cm.rainbow(np.linspace(0, 1, len(slist)))

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
for i, s in enumerate(slist):
    ax.axvline(x = s/n, c=colors[i], linestyle='--',)
    ax.plot(theta_array, (n+1)*stats.binom.pmf(s, n, theta_array) , label=f"{s}", c=colors[i])  # Book06_chap20_(29)
plt.legend(ncols = 2)
plt.show()
plt.close()

# binom.pmf(x, K, p)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk5_Ch20_02
# 图 25. 对比先验分布、似然分布、后验分布，α 取不同值时
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# number of animal samples:
n = 200
# number of rabbits in data:
s = 60

# probability of rabbits among animals:
p = s/n
binom_dist = stats.binom(n, p)
mu = binom_dist.mean()
theta_array = np.linspace(0, 1, 500)

# prior distribution  assumption: 1:1 ratio, i.e., alpha = beta in Beta(alpha, beta) distribution
alpha_arrays = [1, 2, 8, 16, 32, 64]
for alpha in alpha_arrays:
    beta = alpha
    prior = stats.beta(alpha, beta)  # Book06_chap20_(23)
    # posterior distribution
    posterior = stats.beta(s + alpha, n - s + beta)  # Book06_chap20_(34)
    fig, ax = plt.subplots(figsize = (12,6))
    ax.plot(theta_array, prior.pdf(theta_array), label='Prior', c='b')
    ax.plot(theta_array, posterior.pdf(theta_array), label='Posterior', c='r') # 最大化后验概率 MAP

    # factor_normalize = stats.binom(n, theta_array).pmf(s).sum()*1/500
    factor_normalize = 1/(n + 1)
    # note: multiplication factor normalize to normalize likelihood distribution
    ax.plot(theta_array, stats.binom(n, theta_array).pmf(s)/factor_normalize, label='Likelihood', c='g')  # Book06_chap20_(29)

    # Prior mode
    try:
        ax.axvline((alpha-1)/(alpha+beta-2), c='b', linestyle='--', label='Prior mode')
    except:
        pass
    # MAP
    ax.axvline((s+alpha-1)/(n+alpha+beta-2), c='r', linestyle='--', label='MAP')
    # MLE
    ax.axvline(mu/n, c='g', linestyle='--', label='MLE')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 15])
    ax.set_yticks([0,5,10,15])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()
    plt.close()


#%% 3D visualizations Prior distributions
# 图 26. 先验分布，α 取不同值时
a_list = np.arange(1, 64 + 1)
theta_MAP = (s + a_list - 1)/(n + 2*a_list - 2)
theta_MLE = s/n

Prior_PDF_matrix = []

for a_idx in a_list:
    print(s + a_idx)
    print(n - s + a_idx)
    print('=================')
    posterior = stats.beta(a_idx, a_idx)
    pdf_idx = posterior.pdf(theta_array)
    Prior_PDF_matrix.append(pdf_idx)

Prior_PDF_matrix = np.array(Prior_PDF_matrix)
tt, aa = np.meshgrid(theta_array, a_list)
fig, ax = plt.subplots(figsize=(10,10))
ax.contourf(tt, aa, Prior_PDF_matrix, levels = np.linspace(0, Prior_PDF_matrix.max()*1.2, 10), cmap = 'Blues')

# plt.plot(theta_MAP,a, color = 'k')
# plt.axvline(x = theta_MLE, color = 'k')
# prior mode
ax.axvline(x = 0.5, color = 'k')

ax.set_xlabel('Theta')
ax.set_ylabel('Alpha')
ax.set_xlim(0,1)
ax.set_ylim(1,a_list.max())
ax.set_yticks([1,2,8, 16, 32, 64])
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection': '3d'})
tt, aa = np.meshgrid(theta_array, a_list)
ax.plot_wireframe(tt, aa, Prior_PDF_matrix, color = [0,0,0], linewidth = 0.25, rstride=3, cstride=0)
ax.contour(theta_array,a_list, Prior_PDF_matrix, levels = np.linspace(0,Prior_PDF_matrix.max()*1.2,10), cmap = 'Blues')
ax.set_proj_type('ortho')

ax.set_xlabel('Theta')
ax.set_ylabel('Alpha')
ax.set_xlim(0,1)
ax.set_ylim(1,a_list.max())
ax.set_yticks([1,2,8, 16, 32, 64])

ax.set_zlim3d([0,30])
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
ax.grid(False)
plt.show()
plt.close()

#%% Posterior distribution 图 27. 后验分布，α 取不同值时
Prior_PDF_matrix = []
for a_idx in a_list:
    print(s + a_idx)
    print(n - s + a_idx)
    print('=================')
    posterior = stats.beta(s + a_idx, n - s + a_idx)
    pdf_idx = posterior.pdf(theta_array)
    Prior_PDF_matrix.append(pdf_idx)
Prior_PDF_matrix = np.array(Prior_PDF_matrix)

fig, ax = plt.subplots(figsize=(10,10))
ax.contourf(theta_array, a_list, Prior_PDF_matrix, levels = np.linspace(0,Prior_PDF_matrix.max()*1.2,10), cmap = 'Blues')
ax.plot(theta_MAP,a_list, color = 'k')
ax.axvline(x = theta_MLE, color = 'k')
# prior mode
ax.axvline(x = 0.5, color = 'k')

ax.set_xlabel('Theta')
ax.set_ylabel('Alpha')
ax.set_xlim(0,1)
ax.set_ylim(1,a_list.max())
ax.set_yticks([1,2,8, 16, 32, 64])
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': '3d'})
tt, aa = np.meshgrid(theta_array,a_list)
ax.plot_wireframe(tt, aa, Prior_PDF_matrix, color = [0,0,0], linewidth = 0.25, rstride=3, cstride=0)
ax.contour(theta_array, a_list, Prior_PDF_matrix, levels = np.linspace(0,Prior_PDF_matrix.max()*1.2,10), cmap = 'Blues')
ax.set_proj_type('ortho')
ax.set_xlabel('Theta')
ax.set_ylabel('Alpha')
ax.set_xlim(0,1)
ax.set_ylim(1,a_list.max())
ax.set_yticks([1,2,8, 16, 32, 64])
ax.set_zlim3d([0,30])
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
ax.grid(False)
plt.show()
plt.close()































































































































































































































































































