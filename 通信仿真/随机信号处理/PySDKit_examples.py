#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:41:13 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkxODQ0MTQzMg==&mid=2247491408&idx=1&sn=c1229df8b9177cfb2dff1febd5a6eecd&chksm=c034c33d1e02d737f82346635aa70070a075e161f2e9647a8a31dc6d162ede8f15d6ebfe9a65&mpshare=1&scene=1&srcid=0718f9KiB0lEWbtzKLpjL1qB&sharer_shareinfo=0a24e075c981f7531544222adb7f57be&sharer_shareinfo_first=55e9cd3972ad5f935509f96eaaea9335&exportkey=n_ChQIAhIQ6eMFYWkhuEdtoBvMHuR7lBKfAgIE97dBBAEAAAAAANZ0Ghbbd2MAAAAOpnltbLcz9gKNyK89dVj0gB6z1QEBJwoZRf9Df2G0vMgaNeCaXMs6w4GPrMUTpkIuR6wt0ySjRlPGSvLG4BJlOyTOFQIAuy2FjlYZQkAyY%2FpfUCFpocyNBR6DVUVmu0Fu4WmF%2BTEvp6XDKdbrspvEPblCX8jTecsiBBkyr91xHRxX0JjaDyrcDXNLJVhYDiQ6hRSyBzryAFCBEeNXnd5%2F0NScfffOueDtUkolA4uE85Xp2A0Q14JkylllT8at6zkkTfVWdclikLLrdaqZbFV%2Bk5fy5%2BtWVdr8altNWNaH5gi26vh27xNkKKsBomajE6BqFz5Qdl15k2ZSNhYBbPGFZIbihTpom91E&acctmode=0&pass_ticket=LIvd7b5vQ6hTac0SUdIM3NGwOJN7JxGvAOOhrHbftATZACuWJUAnvW7Qzp89%2Fd5y&wx_header=0#rd


"""

from pysdkit.data import test_univariate_signal
from pysdkit.plot import plot_IMFs
from pysdkit import EMD


#%%%%%%%%%%%%%%%%%%%%%%%%% 一元信号数据
# 我们使用以下代码导入信号分解算法的实例并执行算法和可视化：

# 创建用于测试的实例
time, signal = test_univariate_signal(case=1)

# 创建EMD算法的信号分解对象并传入参数
emd = EMD(max_imfs=3)

# 执行信号分解算法
IMFs = emd.fit_transform(signal)
# 或 IMFs = emd(signal)

# 打印信号的尺度大小
print(IMFs.shape)

# 对分解的结果进行可视化
plot_IMFs(signal, IMFs)

# 通过以下代码我们可以创建并调用变分模态分解算法：
from pysdkit.data import test_univariate_signal
from pysdkit.plot import plot_IMFs
from pysdkit import VMD

# 创建用于测试的实例
time, signal = test_univariate_signal(case=1)

# 创建VMD算法的信号分解对象并传入参数
vmd = VMD(alpha=800,
          K=3,  # 待分解模态的数目
          tau=0.0)

# 执行信号分解算法
IMFs = vmd.fit_transform(signal)
# 或 IMFs = vmd(signal)

# 打印信号的尺度大小
print(IMFs.shape)

# 对分解的结果进行可视化
plot_IMFs(signal, IMFs)


# 我们可以进一步在频域中观测该信号中具有不同频率和振幅分量的分离情况：
from pysdkit.plot import plot_IMFs_amplitude_spectra

# frequency domain visualization
plot_IMFs_amplitude_spectra(IMFs, smooth="exp")   # use exp smooth

#%%%%%%%%%%%%%%%%%%%%%%%%% 多元信号数据

from pysdkit.data import test_multivariate_signal
from pysdkit.plot import plot_signal, plot_IMFs
from pysdkit import MVMD

# 创建用于测试的多元信号样例
time, signal = test_multivariate_signal(case=4)

# 实例化算法对象并执行信号分解
mvmd = MVMD(K=2, alpha=2000, tau=0.0)
IMFs = mvmd.fit_transform(signal)

# 对分解得到的多元信号进行可视化分析
plot_IMFs(signal, IMFs, return_figure=True)


#%%%%%%%%%%%%%%%%%%%%%%%%% 二维图像数据

from pysdkit import VMD2D
from pysdkit.data import test_grayscale
from pysdkit.plot import plot_grayscale_image

# 创建用于测试的图像数据
image = test_grayscale()

# 打印并检验图像的尺寸
print(image.shape)

# 实例化用于二维图像的VMD算法
vmd2d = VMD2D(
    K=5, alpha=5000, tau=0.25, DC=True, init="random", tol=1e-6, max_iter=3000
)
# 执行信号分解算法
IMFs = vmd2d.fit_transform(image)
print(IMFs.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%



































































































