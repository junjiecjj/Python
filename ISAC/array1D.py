#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import scipy.constants as CONSTANTS

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#%% Basic Electromagnetic Parameters
Frequency = 10e9
Lightspeed = CONSTANTS.c
Wavelength = Lightspeed/Frequency
Wavenumber = 2 * np.pi/Wavelength

#%% Array Parameters
N = 12
A = np.ones(N)
theta0 = math.radians(30)
# wt = A * np.ones(N, )   # 权重向量
wt = A * np.exp(-1j * (np.pi * np.arange(N) * np.sin(theta0)) )
alpha = np.zeros(N, )


#%% cheb array
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.chebwin.html
# A = np.ones(N)
# wt = A * scipy.signal.windows.chebwin(N, 46)

#%% taylor array
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.taylor.html
# wt = A * scipy.signal.windows.taylor(N, nbar = 4, sll = 10,)

#%% ArrayFactor Samping
Ns = 1000                  # Sampling number
theta = np.linspace(-90, 90, Ns)
Ptheta = np.zeros(Ns, )
mini_a = 1e-5
for num in range(Ns):
    rad = math.radians(theta[num])
    Atheta = np.exp(-1j * (np.pi * (np.arange(N) + 1) * np.sin(rad)) + alpha )  # 导向/方向矢量
    Ptheta[num] = np.abs(wt @ Atheta.T.conjugate()) + mini_a
    # Ptheta[num] = np.abs(np.sum(wt * Atheta.T.conjugate())) + mini_a

dbP = 20 * np.log10(Ptheta)
peaks, _ =  scipy.signal.find_peaks(dbP)



#%% 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(theta, dbP, color='b', linestyle='-', lw = 3, label='',  )
axs.plot(theta[peaks], dbP[peaks], linestyle='', marker = 'o', color='r', markersize = 12)

# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel( r"$\theta(^\circ)$", fontproperties=font2,   ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('Amplitude(dB)', fontproperties=font2,  )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
# legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

x_major_locator = MultipleLocator(20)               #把x轴的刻度间隔设置为1，并存在变量里
axs.xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs.tick_params(direction='in', axis='both', top=True, right=True,labelsize=16, width=3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
axs.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


# plt.show()




#%% https://blog.csdn.net/qq_23176133/article/details/120056777
import math
import cmath
import matplotlib.pyplot as plt
import numpy as np
class Pattern:
    def radiation(self):
        # 单元数量，频率（GHz），位置（mm），幅度，相位（°）
        n_cell = 9
        f = 1.575
        position = [0, 94, 206, 281, 393, 475, 587, 683, 785]
        power = [0.2, 0.8, 0.4, 0.3, 1, 0.9, 0.2, 0.7, 0.4]
        phase = [0, 82, 165, 201, 247, 229, 262, 305, 334]
        # 单元方向图
        data_x = np.arange(-180,180,1)
        data_y = np.cos(data_x/180*np.pi)
        mini_a = 1e-5
        # 2*pi/lamuda
        k = 2 * math.pi * f / 300
        data_new = []
        # 方向图乘积定理
        for i in range(0, len(data_x)):
            a = complex(0, 0)
            k_d = k * math.sin(data_x[i] * math.pi / 180)
            for j in range(0, n_cell):
                a = a + power[j] * data_y[i] * cmath.exp(complex(0,(phase[j] * math.pi / 180 + k_d * position[j])))
            data_new.append(10*math.log10(abs(a)+mini_a))
        plt.plot(data_x, data_new,"y")
        plt.show()
def main(argv=None):
    pattern = Pattern()
    pattern.radiation()

# if __name__ == '__main__':
#     main( )



















































