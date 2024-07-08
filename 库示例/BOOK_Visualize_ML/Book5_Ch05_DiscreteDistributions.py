




#%% Bk5_Ch05_01
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint

a = 1
b = 6

x = np.arange(a, b+1)

discrete_uniform = randint(a, b+1)

p_x = discrete_uniform.pmf(x)

E_x = np.sum(p_x*x)

fig, ax = plt.subplots()

plt.stem(x, p_x)
plt.axvline(x = E_x, color = 'r', linestyle = '--')

plt.xticks(np.arange(a,b+1))
plt.xlabel('x')
plt.ylabel('PMF, $p_X(x)$')
plt.ylim([0,0.2])



#%% Bk5_Ch05_02, 图 4. 圆周率小数点后数字的分布，100 位、1,000 位、10,000 位、100,000 位、1000,000 位

from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt

mp.dps = 1024 + 1
digits = str(mp.pi)[2:]
len(digits)

digits_list = [int(x) for x in digits]

digits_array  = np.array(digits_list)
digits_matrix = digits_array.reshape((32, 32))

# different color
# distribution at different steps

# make a heatmap
import seaborn as sns

fig, ax = plt.subplots()
ax = sns.heatmap(digits_matrix, vmin=0, vmax=9, cmap="RdYlBu_r", yticklabels=False, xticklabels=False)
ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False)


num_digits_array = [100,1000,10000,100000,1000000]
for num_digits in num_digits_array:
    mp.dps = num_digits + 1
    digits = str(mp.pi)[2:]
    len(digits)

    digits_list = [int(x) for x in digits]
    digits_array  = np.array(digits_list)
    fig, ax = plt.subplots()
    counts = np.bincount(digits_array)
    ax.barh(range(10), counts, align='center', edgecolor = [0.8,0.8,0.8])

    for i, v in enumerate(counts):
        ax.text(v + num_digits/400, i, str(v), color='k', va='center')

    ax.axvline(x = num_digits/10, color = 'r', linestyle = '--')
    plt.yticks(range(10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Digit')







#%% Bk5_Ch05_03

from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

KK = [1,2,3,4,8,16,32,64]
p = 0.7 # 0.5

for K in KK:
    x = np.arange(0, K + 1)
    p_x= binom.pmf(x, K, p)
    E_x = np.sum(p_x*x)
    fig, ax = plt.subplots()
    plt.stem(x, p_x)
    plt.axvline(x = E_x, color = 'r', linestyle = '--')

    plt.xticks(np.arange(K+1))
    plt.xlabel('X = x')
    plt.ylabel('PMF, $p_X(x)$')
    plt.ylim([0,p +0.05])






#%% Bk5_Ch05_04,

from scipy.stats import multinomial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

num = 8
x1_array = np.arange(num + 1)
x2_array = np.arange(num + 1)

xx1, xx2 = np.meshgrid(x1_array, x2_array)

xx3 = num - xx1 - xx2
xx3 = np.where(xx3 >= 0.0, xx3, np.nan)

def heatmap_sum(data,i_array,j_array,title,vmin,vmax,cmap,annot = False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(data,cmap= cmap, #'YlGnBu', # YlGnBu
                     cbar_kws={"orientation": "horizontal"},
                     yticklabels=i_array, xticklabels=j_array,
                     ax = ax, annot = annot,
                     linewidths=0.25, linecolor='grey',
                     vmin = vmin, vmax = vmax,
                     fmt = '.3f')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.title(title)
    plt.yticks(rotation=0)


### calculate multinomial probability
p_array = [0.6, 0.1, 0.3]
p_array = [0.3, 0.4, 0.3]
p_array = [0.1, 0.6, 0.3]

PMF_ff = multinomial.pmf(x=np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])).T, n = num, p=p_array)
PMF_ff = np.where(PMF_ff > 0.0, PMF_ff, np.nan)
PMF_ff = np.reshape(PMF_ff, xx1.shape)

### save to excel file

df = pd.DataFrame(np.flipud(PMF_ff))
filepath = 'PMF_ff.xlsx'
# df.to_excel(filepath, index=False)

################################### 3D/2D scatter plot

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
# 图 10. 多项分布 PMF 三维和平面散点图，n = 8，p1 = 0.6，p2 = 0.1，p3 = 0.3
ax.scatter3D(xx1.ravel(), xx2.ravel(), xx3.ravel(), s = 400, marker='.', c = PMF_ff.ravel(), cmap = 'RdYlBu_r')

# ax.contour(xx1, xx2, PMF_ff, 15, zdir='z', offset=0, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xticks([0,num])
ax.set_yticks([0,num])
ax.set_zticks([0,num])

ax.set_xlim(0, num)
ax.set_ylim(0, num)
ax.set_zlim3d(0, num)
ax.view_init(azim=20, elev=20)
# ax.view_init(azim=-30, elev=20)
# ax.view_init(azim=-90, elev=90)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))

ax.grid()
plt.show()

############################## heatmap
# 图 11. 多项分布 PMF 热图和火柴梗图，n = 8，p1 = 0.6，p2 = 0.1，p3 = 0.3
title = 'PMF of binomial distribution'
heatmap_sum(PMF_ff,x1_array,x2_array,title,0,0.12,'plasma_r',True)

###3D stem chart
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')

ax.stem(xx1.ravel(), xx2.ravel(), PMF_ff.ravel(), basefmt=" ")

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('PMF')

ax.set_xlim((0,8))
ax.set_ylim((0,8))
ax.set_zlim((0,0.12))
# ax.set_zticks([])
# ax.grid(False)
ax.view_init(azim=-100, elev=20)
ax.set_proj_type('ortho')
plt.show()

# test only

# print(multinomial.pmf(x=(5,2,1), n=num, p=p_array))




#%% Bk5_Ch05_05
# 图 16. 泊松分布概率质量函数随 λ 变化
from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

x_array = np.arange(0,20 + 1)


# PMF versus x as lambda varies

fig, ax = plt.subplots()

for lambda_ in [1,2,3,4,5,6,7,8,9,10]:
    plt.plot(x_array, poisson.pmf(x_array, lambda_), marker = 'x',markersize = 8, label = '$\lambda$ = ' + str(lambda_))

plt.xlabel('x')
plt.ylabel('PMF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(0,x_array.max())
plt.ylim(0,0.4)
plt.xticks([0,5,10,15,20])
plt.legend()


#%% Bk5_Ch05_06, 图 17. 几何分布概率质量函数 PMF 和 CDF，p = 0.5
from scipy.stats import geom
import matplotlib.pyplot as plt
import numpy as np

p = 0.5

mean,var,skew,kurt = geom.stats(p,moments='mvsk')
print('Expectation, Variance, Skewness, Kurtosis: ', mean, var, skew, kurt)

k_range = np.arange(1,15 + 1)
# PMF versus x

fig, ax = plt.subplots()

plt.stem(k_range, geom.pmf(k_range, p))

plt.xlabel('x')
plt.ylabel('PMF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(1,k_range.max())
plt.ylim(0,p)

# CDF versus x
fig, ax = plt.subplots()

plt.stem(k_range, geom.cdf(k_range, p))
plt.axhline(y = 1, color = 'r', linestyle = '--')

plt.xlabel('x')
plt.ylabel('CDF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(1,k_range.max())
plt.ylim(0,1)



# PMF versus x as p varies, 图 18. 几何分布概率质量函数 PMF 随 p 变化
k_range = np.arange(1,16)

fig, ax = plt.subplots()

for p in [0.4,0.5,0.6,0.7,0.8]:
    plt.plot(k_range, geom.pmf(k_range, p), marker = 'x',markersize = 12, label = 'p = ' + str(p))

plt.xlabel('x')
plt.ylabel('PMF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(1,k_range.max())
plt.ylim(0,0.8)
plt.xticks([1,5,10,15])
plt.legend()




#%% Bk5_Ch05_07,
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
import numpy as np

N = 50 # total number of animals
K = 15 # number of rabbits among N
n = 20 # number of draws without replacement

hyper_g = hypergeom(N, K, n)

x_array = np.arange(np.maximum(0, n + K - N), np.minimum(K,n) + 1)

pmf_rabbits = hyper_g.pmf(x_array)
# 图 20. 超几何分布概率质量函数，N = 50，K = 15，n = 20
fig, ax = plt.subplots()
plt.stem(x_array, pmf_rabbits)

plt.xlabel('x')
plt.ylabel('PMF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(x_array.min(),x_array.max())
plt.ylim(0,pmf_rabbits.max())
plt.xticks([0,5,10,15])



#%% Bk5_Ch05_08, 图 21. 超几何分布 PMF 和二项分布 PMF 关系
from scipy.stats import hypergeom, binom
import matplotlib.pyplot as plt
import numpy as np

p = 0.3  # percentage of rabbits in the population

# N: total number of animals
for N in [100,200,400,800]:
    K = N*p  # number of rabbits among N
    n = 20   # number of draws without replacement

    hyper_g = hypergeom(N, K, n)
    x_array = np.arange(np.maximum(0,n + K - N), np.minimum(K,n) + 1)
    pmf_binom   = binom.pmf(x_array, n, p)
    pmf_hyper_g = hyper_g.pmf(x_array)

    fig, ax = plt.subplots()
    plt.plot(x_array, pmf_hyper_g, '-bx')
    plt.plot(x_array, pmf_binom, '--rx')

    plt.xlabel('x')
    plt.ylabel('PMF')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xlim(x_array.min(),x_array.max())
    plt.ylim(0,0.225)
    plt.xticks([0,5,10,15])


















































































































































































































































































































