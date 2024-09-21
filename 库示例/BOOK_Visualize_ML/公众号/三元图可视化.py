#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:53:51 2024

@author: jack


https://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247486666&idx=1&sn=bbf36c621d333e2263ab19a238f5431d&chksm=c22f25808492ba82d302373b9acf2e79c2e322b32f2c4a3527d90380111b47172c34d20ac300&mpshare=1&scene=1&srcid=0916C538zqDddemYDVhOPbTq&sharer_shareinfo=b568983f12459b351182d0f92b4609bf&sharer_shareinfo_first=b568983f12459b351182d0f92b4609bf&exportkey=n_ChQIAhIQos7fEUw5lco2fWQBk5xCfxKfAgIE97dBBAEAAAAAAMU5FD8B5ooAAAAOpnltbLcz9gKNyK89dVj0Qaiu0T5AJTSKAPd10lePgIrTWh0uF6a%2B7%2Fd9qBMdUWCFj8flJ4PpBU2rqDON1RHXOD66BfqZbgD29PXXZa1mrBK1Uw2CVkR3i7uvBrHkT%2F%2B38SIgxrprQ1kaB8voHsjje6l%2B0RMEdQNA0NKRV%2FCUivMVNkodfXTVuQHpuuy7urxRZKHh1VMvRrmUVcmy2d0JcZwZdRQ1tgfJlgU4qOwP2724vzE85s%2FO3x705TBoi2qZ0oChJLbCYbr3sEjiCVnQbLKZXoRkue%2BPtS%2F%2F8wkxCUKFZ%2Fm2eKR%2FSsr%2FvdUlznegKRA%2BZay53cS9AOQs6uXNN54DHL3scszO&acctmode=0&pass_ticket=BWewIwspFdxzXTY4iL9mQ7lp8S6Uttm7xTk6ZcH2UAV9FENBDldeKgnK4AvPDki5&wx_header=0#rd

https://mpltern.readthedocs.io/en/latest/gallery/index.html



"""



import matplotlib.pyplot as plt
from mpltern.datasets import get_dirichlet_pdfs
import pandas as pd
# 定义 alpha 参数
alpha = (2.0, 4.0, 8.0)
# 获取 Dirichlet 分布的概率密度函数 (PDF) 数据
t, l, r, v = get_dirichlet_pdfs(n=61, alpha=alpha)
df = pd.DataFrame({
    'x1': t,
    'x2': l,
    'x3': r,
    'pdf': v
})
# df


# 绘制三元图
fig = plt.figure(figsize=(8, 6), )

ax = fig.add_subplot(1, 1, 1, projection="ternary")
# 设置颜色映射和阴影
cmap = "Blues"
shading = "gouraud"
# 绘制三元图的颜色图
cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)
# 添加等高线
ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)
ax.set_tlabel("$x_1$")
ax.set_llabel("$x_2$")
ax.set_rlabel("$x_3$")
ax.set_title("${\\mathbf{\\alpha}}$=" + str(alpha))
cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(cs, cax=cax)
colorbar.set_label("PDF", rotation=270, va="baseline")
# plt.savefig("1.pdf", format='pdf', bbox_inches='tight')
plt.show()


# 不同条件下的三元图对比
fig = plt.figure(figsize=(10.8, 8.8), )
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
# 定义 alpha 参数
alphas = ((1.5, 1.5, 1.5), (5.0, 5.0, 5.0), (1.0, 2.0, 2.0), (2.0, 4.0, 8.0))

# 循环绘制四个不同的三元图
for i, alpha in enumerate(alphas):
    ax = fig.add_subplot(2, 2, i + 1, projection="ternary")

    # 获取 Dirichlet 分布的概率密度函数 (PDF) 数据
    t, l, r, v = get_dirichlet_pdfs(n=61, alpha=alpha)

    # 设置颜色映射和阴影
    cmap = "Blues"
    shading = "gouraud"

    # 绘制三元图的颜色图
    cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)

    # 添加等高线
    ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)

    # 设置标签
    ax.set_tlabel("$x_1$")
    ax.set_llabel("$x_2$")
    ax.set_rlabel("$x_3$")

    # 设置标题
    ax.set_title("${\\mathbf{\\alpha}}$=" + str(alpha))

    # 插入颜色条
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax)
    colorbar.set_label("PDF", rotation=270, va="baseline")
# plt.savefig("2.pdf", format='pdf', bbox_inches='tight')
plt.show()












