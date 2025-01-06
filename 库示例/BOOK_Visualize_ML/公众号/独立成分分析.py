#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:13:26 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488301&idx=1&sn=94c5a0092060ac65dcc1392afa164b37&chksm=c13a89aafa04059d93a72ff7c1135dd1dd988ed123761d2cb6e0e454dc1dec211a8c588039eb&mpshare=1&scene=1&srcid=0104BQKR70hAbmI1pCNjJh9l&sharer_shareinfo=ad408821e1ad695ee0e3dfb5ef5ec5a6&sharer_shareinfo_first=ad408821e1ad695ee0e3dfb5ef5ec5a6&exportkey=n_ChQIAhIQ2Oi6HxgAh570WU5%2Bsz%2FFZRKfAgIE97dBBAEAAAAAAFkMGK9jtiUAAAAOpnltbLcz9gKNyK89dVj0cCX8BhYO8zkCOdSiP5lr4WRDIAJOobcoHziZg3pbKPwnYPEWCYRe98ypp116ddow7KZbUG2JnyYHVsKHA2ng3C%2FpIwRp2COfMorgN7PyLhoswnNWmFgzJ9Jqvxi9Z3tqFaKXiNF3sAmY%2BShpfKwW%2B6uKNjjc3AAmq8kYNhUXFv9bD%2Bw0UwnMCCkvTrHHHzc5IrGtv8bqjIvbjtyaxXxigMlT5ShGyi2vQT37u%2F9sFUDZqF1qWQlFosXNdF1wEiuhGGKmkGxinZrhqJQmj7Ctllmx%2FEqr63SRCmeWaqL8eFpFi1N%2F7MR%2FRM8YrzZr2gPwEI0V7jFnkla6&acctmode=0&pass_ticket=7hVhqqUN5vee1hAnZ%2BapQMKxI9ehV%2B3l0hQ5HPuRJU9skfCHeiOxd0t8vHz9kW4R&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 第一步：生成模拟EEG信号的独立源信号
np.random.seed(42)
n_samples = 2000  # 样本点数
time = np.linspace(0, 8, n_samples)  # 时间轴，模拟8秒

# 生成三个独立信号（Independent Components）
s1 = np.sin(2 * time)                     # 正弦信号
s2 = np.sign(np.sin(3 * time))            # 方波信号
s3 = np.random.normal(size=n_samples)    # 高斯噪声信号

# 将信号标准化，以确保它们具有单位方差
S = np.c_[s1, s2, s3]
S /= S.std(axis=0)

# 第二步：创建一个混合矩阵，并生成观测信号（混合信号）
A = np.array([[1, 1, 0.5], [0.5, 2, 1], [1.5, 1, 2]])  # 混合矩阵
X = S.dot(A.T)  # 混合信号，通过矩阵乘法实现

# 第三步：使用ICA分离混合信号
ica = FastICA(n_components=3, random_state=42)  # 创建ICA模型
S_ica = ica.fit_transform(X)  # 通过ICA分离出的信号
A_ica = ica.mixing_  # ICA估计的混合矩阵（可用于验证）

# 第四步：绘制结果对比
plt.figure(figsize=(15, 16))  # 设置画布大小

# 原始独立信号
plt.subplot(4, 1, 1)  # 四行一列的第一个子图
plt.title("Original Independent Signals (Sources)", fontsize=14)
colors = ['red', 'blue', 'green']
for i, signal in enumerate(S.T):
    plt.plot(time, signal + i * 3, color=colors[i], label=f"Source {i+1}")  # 每个信号在y轴偏移，便于区分
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# 混合信号（观测信号）
plt.subplot(4, 1, 2)  # 四行一列的第二个子图
plt.title("Mixed Signals (Observed)", fontsize=14)
colors = ['purple', 'orange', 'cyan']
for i, signal in enumerate(X.T):
    plt.plot(time, signal + i * 3, color=colors[i], label=f"Mixed {i+1}")  # 同样对信号偏移
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# 分离后的独立信号
plt.subplot(4, 1, 3)  # 四行一列的第三个子图
plt.title("Recovered Independent Signals (ICA Output)", fontsize=14)
colors = ['magenta', 'lime', 'gold']
for i, signal in enumerate(S_ica.T):
    plt.plot(time, signal + i * 3, color=colors[i], label=f"Recovered {i+1}")  # 偏移便于区分
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# 混合矩阵的热力图可视化
plt.subplot(4, 1, 4)  # 四行一列的第四个子图
plt.title("Heatmap of Mixing Matrix", fontsize=14)
plt.imshow(A, cmap='viridis', aspect='auto')
plt.colorbar(label='Value')
plt.xticks(range(A.shape[1]), labels=[f"Source {i+1}" for i in range(A.shape[1])])
plt.yticks(range(A.shape[0]), labels=[f"Mixed {i+1}" for i in range(A.shape[0])])
plt.xlabel("Source Components")
plt.ylabel("Mixed Signals")

# 调整布局并展示
plt.tight_layout()
plt.show()




# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488019&idx=1&sn=23903ce599082f1fa29c8a7644cd39b4&chksm=c106c7917a44e4e18934f76519027524d9c2c291f5df514e49a2b692f2667d4260d71ddaf3cb&mpshare=1&scene=1&srcid=0104fS5QKBqt1fxgfRezvXhn&sharer_shareinfo=50eaf452ab5c7c1ec727c58b5a367941&sharer_shareinfo_first=50eaf452ab5c7c1ec727c58b5a367941&exportkey=n_ChQIAhIQiwPz4DJCciC2nhLVsb8Q6hKfAgIE97dBBAEAAAAAAPgCJ5QPW6kAAAAOpnltbLcz9gKNyK89dVj0o7sdBaRfrNT3MeSngeNYyqIqYVOqPz1NUMmZ%2FxAyKlVNkGeHvIGbapQh8561op03AyWdn0GLuV9yGtUWf0hVrZY9CYJLmkhHLgS%2ByyOl%2BRRZ3PvazBXfz7P0eo%2BsbOebknTQpslCtD0E4laV%2BOYiswsPiV184y0XnLm%2B6%2FPzhD1OZP6MaCbc7D%2FH%2Bv6RxDYOyfCIUi%2BX4dn1rKXSd3PIvy%2B6y6CFNsgXXvjCdKTT9WfNYqI%2F3P4PkwBZUZocJT5ovhUCC1nYNvcBv744NrFYBNsjf5EjCDW8c6WHsyc1sfempXPZD0kkIA2OGk3kOmWASf2YiNSfIu%2FQ&acctmode=0&pass_ticket=KV0STtUIS2TFmZ3Oac9ZemlP8VlhWl%2FBN7wRiu8Ef6%2FsXDcptmbqzoM6SMW9FJoA&wx_header=0#rd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 设置随机种子
np.random.seed(0)

# 生成虚拟信号数据
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 生成3个独立的信号：正弦波、方波和噪声
s1 = np.sin(2 * time)  # 正弦波
s2 = np.sign(np.sin(3 * time))  # 方波
s3 = np.random.normal(size=n_samples)  # 高斯噪声

# 将信号组合成矩阵
S = np.c_[s1, s2, s3]

# 将信号标准化到范围内
S /= S.std(axis=0)

# 生成混合数据（线性混合）
A = np.array([[1, 1, 0.5], [0.5, 2, 1], [1.5, 1, 2.5]])  # 混合矩阵
X = np.dot(S, A.T)  # 生成混合信号

# 使用FastICA从混合信号中分离独立成分
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # 重构后的信号
A_ = ica.mixing_  # 分离出的混合矩阵

# 绘制原始信号、混合信号和分离出的信号
plt.figure(figsize=(15, 10))

# 原始信号
plt.subplot(3, 1, 1)
plt.title("Original Signals")
colors = ['red', 'blue', 'green']
for i, signal in enumerate(S.T):
    plt.plot(time, signal, color=colors[i], label=f"Signal {i+1}")
plt.legend(loc='upper right')

# 混合信号
plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
colors = ['orange', 'purple', 'brown']
for i, signal in enumerate(X.T):
    plt.plot(time, signal, color=colors[i], label=f"Mixed {i+1}")
plt.legend(loc='upper right')

# 分离出的信号
plt.subplot(3, 1, 3)
plt.title("ICA Recovered Signals")
colors = ['cyan', 'magenta', 'yellow']
for i, signal in enumerate(S_.T):
    plt.plot(time, signal, color=colors[i], label=f"Recovered {i+1}")
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488850&idx=1&sn=d3533dc33db7843fe8dee0be659afad1&chksm=c16ff99f18f19f7c7fb75d8ad8ed174e4c9727a6c5fa706f1686371f3fa5d1ed629ce8bdddb8&mpshare=1&scene=1&srcid=01041RiX8zCqb8uBLR9t2poQ&sharer_shareinfo=e7288ca74f8c24842f71d52bfc1a8d62&sharer_shareinfo_first=e7288ca74f8c24842f71d52bfc1a8d62&exportkey=n_ChQIAhIQRv%2B7g5tmOo2tH4MvJf2RgBKfAgIE97dBBAEAAAAAAMx7Lk6hjVsAAAAOpnltbLcz9gKNyK89dVj0bTbdNRQvFfa42QBM7mhb%2BFZulfQ%2B8pxfc%2BZjyLjDCDc8M7xmGJu2Zc5KDvR3bnHGUGVi9gARgvyrhvlJ9gNWtTA%2FK0zGc1BNC3lYAK3BCqayBgMbI8Zi%2FP1k4bnb66IF2g08gSjD%2BgOTUOGBfbxdtI75XMdFyMZ9rlTFsdnhzQH5SZPA7Yxfpj9S0JNSrIc2IHg6CwRDj3KFqYOCz4uw5QepMp28oPSOnfosD35WJkOn8aK%2FohZoghZaSoG%2BDbsdfWzwCFnu0IU7O%2FCYaZMUYHi1Y%2FBMKd%2FQ4Sz%2BX%2FzF9317eyP%2FZFPXwh5CGvD2%2FD7kpdWwOBQVC3us&acctmode=0&pass_ticket=Ut47H4%2Fttrwc42eVmIt9Lu%2BUZCyJwIpVtqxhmfnOCtBAkI40dpxBIc300U02akrP&wx_header=0#rd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Step 1: 数据生成
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 独立信号生成
s1 = np.sin(2 * time)  # 正弦信号
s2 = np.sign(np.sin(3 * time))  # 方波信号
s3 = np.random.normal(size=n_samples)  # 高斯噪声

# 合并信号
S = np.c_[s1, s2, s3]
S /= S.std(axis=0)  # 标准化

# 混合信号生成
A = np.array([[1, 1, 0.5], [0.5, 2, 1], [1.5, 1, 2]])  # 混合矩阵
X = np.dot(S, A.T)  # 观测信号

# Step 2: 应用 ICA
ica = FastICA(n_components=3, random_state=42)
S_estimated = ica.fit_transform(X)  # 分离出的信号
A_estimated = ica.mixing_  # 估计的混合矩阵

# Step 3: 绘图分析
fig, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)

# 原始独立信号
axes[0, 0].plot(time, S[:, 0], color='red')
axes[0, 0].set_title('Original Signal 1')
axes[0, 1].plot(time, S[:, 1], color='blue')
axes[0, 1].set_title('Original Signal 2')

# 混合信号
axes[1, 0].plot(time, X[:, 0], color='green')
axes[1, 0].set_title('Mixed Signal 1')
axes[1, 1].plot(time, X[:, 1], color='purple')
axes[1, 1].set_title('Mixed Signal 2')

# 分离信号
axes[2, 0].plot(time, S_estimated[:, 0], color='orange')
axes[2, 0].set_title('Recovered Signal 1')
axes[2, 1].plot(time, S_estimated[:, 1], color='cyan')
axes[2, 1].set_title('Recovered Signal 2')

plt.suptitle('Independent Component Analysis (ICA) Results', fontsize=16)
plt.show()





# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488019&idx=1&sn=23903ce599082f1fa29c8a7644cd39b4&chksm=c1f260d446993af8b15013c079e10d9b02da149ad7f357e1922d528ce11d3025c977a611c2e7&mpshare=1&scene=1&srcid=01043YmdiZWHz8JHGIPm4azn&sharer_shareinfo=93a6aa1abc1986e55439bbb5fb217ce9&sharer_shareinfo_first=93a6aa1abc1986e55439bbb5fb217ce9&exportkey=n_ChQIAhIQsLRCT84rdEq2mb7m3muNphKfAgIE97dBBAEAAAAAACG8JLE9RIkAAAAOpnltbLcz9gKNyK89dVj0Qo4mQqviZeOvws7jlKXAZhRMFjXVRaJotqe8W9RkG0aIRMs6KR2EHkoE7g7RJRBCWKPdmucwbLZuQjgenoewfKzeEcCd8kRxnlNrVIfF6jxjfr39vJyKCK%2FcPqfD2QMUnwf%2BwERuEyCsj1rUZhw2je0J9bTJD0era2T%2B9ViCMJ%2F%2FqyXiScp%2B2M7iZRhd86V7vKoEIlVaFizrYdgA7sgH28%2BJXIZNVPz%2FwEXth7Oby9exhy1m5r2WiwV0wk2q4RF2DNmMhTWK2UnywJCRHC1ERm5I97Ld%2BAOP6Um8pnBWqbe7p0b7FGvphGgSm1dlqEgznykA834WcnBN&acctmode=0&pass_ticket=HGPOjMJbUvRCgQD4GRfBJ1Xvr1VYiVo4yJm7wXR2g5LWltGJVTShpPJyrbKnruYL&wx_header=0#rd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 设置随机种子
np.random.seed(0)

# 生成虚拟信号数据
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 生成3个独立的信号：正弦波、方波和噪声
s1 = np.sin(2 * time)  # 正弦波
s2 = np.sign(np.sin(3 * time))  # 方波
s3 = np.random.normal(size=n_samples)  # 高斯噪声

# 将信号组合成矩阵
S = np.c_[s1, s2, s3]

# 将信号标准化到范围内
S /= S.std(axis=0)

# 生成混合数据（线性混合）
A = np.array([[1, 1, 0.5], [0.5, 2, 1], [1.5, 1, 2.5]])  # 混合矩阵
X = np.dot(S, A.T)  # 生成混合信号

# 使用FastICA从混合信号中分离独立成分
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # 重构后的信号
A_ = ica.mixing_  # 分离出的混合矩阵

# 绘制原始信号、混合信号和分离出的信号
plt.figure(figsize=(15, 10))

# 原始信号
plt.subplot(3, 1, 1)
plt.title("Original Signals")
colors = ['red', 'blue', 'green']
for i, signal in enumerate(S.T):
    plt.plot(time, signal, color=colors[i], label=f"Signal {i+1}")
plt.legend(loc='upper right')

# 混合信号
plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
colors = ['orange', 'purple', 'brown']
for i, signal in enumerate(X.T):
    plt.plot(time, signal, color=colors[i], label=f"Mixed {i+1}")
plt.legend(loc='upper right')

# 分离出的信号
plt.subplot(3, 1, 3)
plt.title("ICA Recovered Signals")
colors = ['cyan', 'magenta', 'yellow']
for i, signal in enumerate(S_.T):
    plt.plot(time, signal, color=colors[i], label=f"Recovered {i+1}")
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()































