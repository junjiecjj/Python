#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:38:19 2024

@author: jack


z = -(x- mu)/beta
f(x) = e^{-z-e^{-z}} * 1/beta
The function has a mean of mu + 0.57721 * beta   and a variance of pi^2 * beta^2 / 6
"""



import numpy as np
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)




#%%=============================================================================
##                  scipy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html
#================================================================================
loc = 0
scale = 2


x = np.linspace(gumbel_r.ppf(0.01, loc = loc, scale = scale),  gumbel_r.ppf(0.99, loc = loc, scale = scale), 100)
ax.plot(x, gumbel_r.pdf(x, loc = loc, scale = scale), 'r-', lw = 5, alpha = 0.6, label = 'gumbel_r pdf')

rv = gumbel_r( loc = loc, scale = scale)
ax.plot(x, rv.pdf(x, ), 'k-', lw=2, label='frozen pdf')

frozencdf = rv.cdf(x)
# frozencdf = gumbel_r( loc = loc, scale = scale).cdf(x, )
cdf = gumbel_r.cdf(x, loc = loc, scale = scale,)

# ax.plot(x, cdf, 'b-', lw = 5, alpha = 0.6, label = 'gumbel_r cdf')
# ax.plot(x, frozencdf, 'm-', lw=2, label='frozen cdf')


vals = gumbel_r.ppf([0.001, 0.5, 0.999], loc = loc, scale = scale)
print(np.allclose([0.001, 0.5, 0.999], gumbel_r.cdf(vals, loc = loc, scale = scale)))


r = gumbel_r.rvs(size=1000, loc = loc, scale = scale)

ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
# ax.set_xlim([x[0], x[-1]])
ax.legend(loc='best', frameon=False)
plt.show()



#%%=============================================================================
##                  numpy
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gumbel.html
#================================================================================

rng = np.random.default_rng()
mu, beta = loc, scale # location and scale
s = rng.gumbel(mu, beta, 10000)

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta) * np.exp( -np.exp( -(bins - mu) /beta) ),  linewidth = 2, color = 'r')
plt.show()



## 产生多个高斯分布序列，并记录每个序列的最大值，然后统计最大值的分布，用Gumble分布拟合最大值分布，用gaussian分布拟合最大值分布.
means = []
maxima = []
for i in range(0, 1000):
   a = rng.normal(mu, beta, 1000)
   means.append(a.mean())
   maxima.append(a.max())
count, bins, ignored = plt.hist(maxima, 30, density=True)


# plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta) * np.exp(-np.exp(-(bins - mu)/beta)),  linewidth=2, color='y')
# plt.plot(bins, 1/(beta * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * beta**2)),  linewidth=2, color='m')

beta1 = np.std(maxima) * np.sqrt(6) / np.pi
mu1 = np.mean(maxima) - 0.57721*beta1
plt.plot(bins, (1/beta1)*np.exp(-(bins - mu1)/beta1) * np.exp(-np.exp(-(bins - mu1)/beta1)), linewidth=2, color='r')
plt.plot(bins, 1/(beta1 * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu1)**2 / (2 * beta1**2)), linewidth=2, color='g')

plt.show()


#%%=============================================================================
##    手动实现 Gumble 分布
# https://blog.csdn.net/jackytintin/article/details/79364490
#===============================================================================
# 概率密度函数（PDF）
def gumbel_pdf(x, mu = 0, beta = 1):
    z = (x - mu) / beta
    return np.exp(-z - np.exp(-z)) / beta

# 累计密度函数（CDF）
def gumbel_cdf(x, mu = 0, beta = 1):
    z = (x - mu) / beta
    return np.exp(-np.exp(-z))


# CDF 的反函数
def gumbel_ppf(y, mu = 0, beta = 1, eps = 1e-20):
    return mu - beta * np.log(-np.log(y + eps))

# 利用反函数法和生成服从 Gumbel 分布的随机数。
def sample_gumbel(shape):
    p = np.random.random(shape)
    return gumbel_ppf(p)


print(gumbel_pdf(0.5, 0.5, 2))     # 0.18393972058572117
print(gumbel_r.pdf(0.5, loc = 0.5, scale = 2))  # 0.18393972058572117

print(gumbel_ppf(gumbel_cdf(5, 0.5, 2), 0.5, 2))  #  5
print(sample_gumbel([2,3]))


# 首先来看常规的采样方法。
def softmax(logits):
    max_value = np.max(logits)
    exp = np.exp(logits - max_value) # 减去最大值，防止指数溢出
    exp_sum = np.sum(exp)
    dist = exp / exp_sum
    return dist

def sample_with_softmax(logits, size):
    pros = softmax(logits)
    return np.random.choice(list(np.arange(10)), size, p = pros)

# 基于 gumbel 的采样（gumbel-max）
def sample_with_gumbel_noise(logits, size):
    noise = sample_gumbel((size, len(logits)))
    # noise = gumbel_r.rvs(size = (size, len(logits)), loc = 0, scale = 1)
    return np.argmax(noise + logits, axis = 1)

# gumbel-max 方法的采样效果等效于基于 softmax 的方式
# np.random.seed(1111)
logits = (np.random.random(10) - 0.5) * 2  # (-1, 1)

pop = 100000
softmax_samples = sample_with_softmax(logits, pop)
gumbel_samples = sample_with_gumbel_noise(logits, pop)

fig, ax = plt.subplots(1, 1)
plt.subplot(1, 2, 1)
plt.hist(softmax_samples)

plt.subplot(1, 2, 2)
plt.hist(gumbel_samples)


#%%=============================================================================
##    手动实现 Gumble 分布
# https://blog.csdn.net/MTandHJ/article/details/117299328?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-117299328-blog-79364490.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-117299328-blog-79364490.235%5Ev43%5Epc_blog_bottom_relevance_base8
#================================================================================
"""
假设我们有一个离散的分布P = [p_1 , p_2 , ⋯   , p_k ]  p_i 表示为第i类的概率, 则从该分布中采样z 等价于:

    z = arg max_{i} [g_i + log(p_i)], g_i ~ Gumbel(0,1), i.i.d
"""


Pi = (np.random.random(10) - 0.5) * 2
Max = Pi.max()
exp = np.exp(Pi - Max)
Pi1 = exp/np.sum(exp)

size = 100000
Pi_samples =  np.random.choice(Pi1.size, size, p = Pi1)

noise = gumbel_r.rvs(size = (size, len(logits)), loc = 0, scale = 1)
Gumbel_samples = np.argmax(noise + np.log(Pi1), axis = 1)
Gumbel_samples2 = np.argmax(noise + Pi, axis = 1)

fig, ax = plt.subplots( figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(Pi_samples)

plt.subplot(1, 3, 2)
plt.hist(Gumbel_samples)

plt.subplot(1, 3, 3)
plt.hist(Gumbel_samples2)
# plt.close()






#%%=============================================================================
##    手动实现 Gumble 分布
# https://blog.csdn.net/weixin_40255337/article/details/83303702?spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-83303702-blog-79364490.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-83303702-blog-79364490.235%5Ev43%5Epc_blog_bottom_relevance_base8
#================================================================================


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

n_cats = 8
n_samples = 1000
cats = np.arange(n_cats)
probs = np.random.randint(low=1, high=20, size=n_cats)
probs = probs / sum(probs)
logits = np.log(probs)

def plot_probs():   # 真实概率分布
    plt.bar(cats, probs)
    plt.xlabel("Category")
    plt.ylabel("Original Probability")

def plot_estimated_probs(samples, ylabel = ''):
    n_cats = np.max(samples)+1
    estd_probs, _, _ = plt.hist(samples, bins=np.arange(n_cats+1), align='left', edgecolor='white')
    plt.xlabel('Category')
    plt.ylabel(ylabel + 'Estimated probability')
    return estd_probs

def print_probs(probs):
    print(probs)

samples = np.random.choice(cats,p=probs,size=n_samples) # 依概率采样

plt.figure()
plt.subplot(1,2,1)
plot_probs()
plt.subplot(1,2,2)
estd_probs = plot_estimated_probs(samples)
plt.tight_layout() # 紧凑显示图片
# plt.savefig('/home/zhumingchao/PycharmProjects/matplot/gumbel1')

print('Original probabilities:\t',end='')
print_probs(probs)
print('Estimated probabilities:\t',end='')
print_probs(estd_probs)
plt.show()
######################################

def sample_gumbel(logits):
    noise = np.random.gumbel(size=len(logits))
    sample = np.argmax(logits+noise)
    return sample
gumbel_samples = [sample_gumbel(logits) for _ in range(n_samples)]

def sample_uniform(logits):
    noise = np.random.uniform(size=len(logits))
    sample = np.argmax(logits+noise)
    return sample
uniform_samples = [sample_uniform(logits) for _ in range(n_samples)]

def sample_normal(logits):
    noise = np.random.normal(size=len(logits))
    sample = np.argmax(logits+noise)
    # print('old',sample)
    return sample
normal_samples = [sample_normal(logits) for _ in range(n_samples)]

plt.figure(figsize=(10,4))
plt.subplot(1,4,1)
plot_probs()
plt.subplot(1,4,2)
gumbel_estd_probs = plot_estimated_probs(gumbel_samples,'Gumbel ')
plt.subplot(1,4,3)
normal_estd_probs = plot_estimated_probs(normal_samples,'Normal ')
plt.subplot(1,4,4)
uniform_estd_probs = plot_estimated_probs(uniform_samples,'Uniform ')
plt.tight_layout()
plt.savefig('/home/zhumingchao/PycharmProjects/matplot/gumbel2')

print('Original probabilities:\t',end='')
print_probs(probs)
print('Gumbel Estimated probabilities:\t',end='')
print_probs(gumbel_estd_probs)
print('Normal Estimated probabilities:\t',end='')
print_probs(normal_estd_probs)
print('Uniform Estimated probabilities:\t',end='')
print_probs(uniform_estd_probs)
plt.show()
#######################################

def softmax(logits):
    return np.exp(logits)/np.sum(np.exp(logits))

def differentiable_sample_1(logits, cats_range, temperature=.1):
    noise = np.random.gumbel(size=len(logits))
    logits_with_noise = softmax((logits+noise)/temperature)
    # print(logits_with_noise)
    sample = np.sum(logits_with_noise*cats_range)
    return sample
differentiable_samples_1 = [differentiable_sample_1(logits,np.arange(n_cats)) for _ in range(n_samples)]

def differentiable_sample_2(logits, cats_range, temperature=5):
    noise = np.random.gumbel(size=len(logits))
    logits_with_noise = softmax((logits+noise)/temperature)
    # print(logits_with_noise)
    sample = np.sum(logits_with_noise*cats_range)
    return sample
differentiable_samples_2 = [differentiable_sample_2(logits,np.arange(n_cats)) for _ in range(n_samples)]

def plot_estimated_probs_(samples,ylabel=''):
    samples = np.rint(samples)
    n_cats = np.max(samples)+1
    estd_probs,_,_ = plt.hist(samples,bins=np.arange(n_cats+1),align='left',edgecolor='white')
    plt.xlabel('Category')
    plt.ylabel(ylabel+'Estimated probability')
    return estd_probs

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
gumbelsoft_estd_probs_1 = plot_estimated_probs_(differentiable_samples_1,'Gumbel softmax')
plt.subplot(1,2,2)
gumbelsoft_estd_probs_2 = plot_estimated_probs_(differentiable_samples_2,'Gumbel softmax')
plt.tight_layout()
plt.savefig('/home/zhumingchao/PycharmProjects/matplot/gumbel3')

print('Gumbel Softmax Estimated probabilities:\t',end='')
print_probs(gumbelsoft_estd_probs_1)
plt.show()





#%%=============================================================================
# https://mathor.blog.csdn.net/article/details/127890859?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-127890859-blog-83303702.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-127890859-blog-83303702.235%5Ev43%5Epc_blog_bottom_relevance_base8&utm_relevant_index=5

# https://rooney.blog.csdn.net/article/details/123014858?spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-123014858-blog-79364490.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-123014858-blog-79364490.235%5Ev43%5Epc_blog_bottom_relevance_base8

# https://blog.csdn.net/qq_40742298/article/details/136516578?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-136516578-blog-79364490.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-136516578-blog-79364490.235%5Ev43%5Epc_blog_bottom_relevance_base8
#================================================================================


































































































































































































































































