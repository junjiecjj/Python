#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:33:04 2025

@author: jack
"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


#%% Program 11.1: scattering function.m: Plot scattering function power delay profile and Doppler spectrum
import sympy
from sympy import symbols, lambdify, expand, simplify
# from sympy.abc import x, y
from matplotlib import cm
fc = 1800e6   #  Carrier frequency (Hz)
fm = 200      #  Maximum Doppler shift (Hz)
trms = 30e-6  #  RMS Delay spread (s)
Plocal = 0.2   # local-mean power (W)

f_delta = fm/30                                      # step size for frequency shifts
f = np.arange((fc - fm) + f_delta, (fc+fm)  , f_delta) # normalized freq shifts
tau = np.arange(0, trms*3 + trms/5, trms/5)            # generate range for propagation delays
TAU, F = np.meshgrid(tau, f)                         # all possible combinations of Taus and Fs

## 1
#Example Scattering function equation
Z = Plocal/(4 * np.pi * fm * np.sqrt(1 - ((F - fc) / fm)**2))*1/trms * np.exp(-TAU/trms)

### 2
# x, y = symbols('x  y')
# f_xy = Plocal/ trms / (4 * np.pi * fm * sympy.sqrt(1 - ((y - fc) / fm)**2)) * sympy.exp(-x/trms)
# f_xy_fcn = lambdify([x, y], f_xy)
# # 将符号函数表达式转换为Python函数
# ff = f_xy_fcn(TAU, F)

###########

fig = plt.figure(figsize=(12, 10), constrained_layout = True)

ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=2,projection = '3d' ) # projection = '3d'
ax1.set_proj_type('ortho')
norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
surf = ax1.plot_wireframe(TAU, (F-fc)/fm, Z, color = '#0070C0', rstride=1, cstride=1, linewidth = 1)

ax1.set_xlabel(r'Delay $\tau$')
ax1.set_ylabel('(f-fc)/fm')
ax1.set_zlabel('Received power')
# 设置X、Y、Z面的背景是白色
ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.view_init(azim=-135, elev=30)
ax1.grid(False)

ax2 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax2.plot(tau, Z[0,:], color="b", linestyle="-.", linewidth=1.0, marker="d")
ax2.set_title("Power Delay Profile", fontsize=20)
ax2.set_xlabel("Delay $\tau$", fontsize=15)
ax2.set_ylabel("Received power", fontsize=15)

# 构建2×3的区域，在起点(1, 0)处开始，跨域1行1列的位置区域绘图
ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
ax3.plot((f-fc)/fm, Z[:,0], color="k",  label=r"$y=\mathrm{sin}{x}$")
# ax3.plot(t, x4, color="r", linestyle="-.", linewidth=1.0, label=r"$y=\mathrm{cos}{3x + \frac{\pi}{3}}$")
ax3.set_title("Doppler Power Spectrum", fontsize=20)
ax3.set_xlabel("Doppler shift (f-fc)/fm", fontsize=15)
ax3.set_ylabel("Received power", fontsize=15)
ax3.legend(fontsize=15)

# 设置整个图的标题
plt.suptitle(r"Scattering function S$(f,\tau)$", fontsize=25)

# plt.savefig("不规则画布.png") # 要想在pgf后端下显示图片，就必须使用该句命令，否则报错
plt.show()



#%% Program 11.2: plot fcf.m: Frequency correlation function (FCF) from power delay profile
Ps = np.array([0, -2.9, -5.8, -8.7, -11.6])             # list of received powers in dBm
TAUs = np.array([0, 50e-9, 100e-9, 150e-9, 200e-9])  # list of propagation delays

p_i = 10**(Ps/10)  # power levels in linear scale

# Frequency Correlation Function (FCF)
nfft = 216  # FFT size
FCF = scipy.fft.fft(p_i, nfft)/(p_i.size) # Take FFT of PDP
FCF = np.abs(FCF)/max(np.abs(FCF))    # normalize
dTau = 50e-9                      # spacing of taus in the PDP
f2 = 1/dTau/2 * np.linspace(0, 1, int(nfft/2+1))  # x-axis frequency bins

##### plot
fig, axs = plt.subplots(2, 1, figsize = (6, 8), constrained_layout = True)

# x
axs[0].stem(TAUs, Ps, bottom = np.min(Ps))
axs[0].set_xlabel(r'Relative delays $\tau$(s)',)
axs[0].set_ylabel(r'p(\tau)',)
axs[0].set_title("Power Delay Profile")
# axs[0].legend()

axs[1].plot(f2, FCF[:int(nfft/2+1)], color = 'r', label = '')
axs[1].set_xlabel(r'Frequency difference $\Delta$f(Hz)',)
axs[1].set_ylabel(r'Correlation $\rho(\Delta f)$',)
axs[1].set_title("Frequency Correlation Function")

plt.show()
plt.close()


#%% Program 11.3, 11.4, 11.5, 11.6
def meas_continuous_PDP(fun, lowerLim, upperLim):
    moment_1 = lambda x: x * fun(x)
    meanDelay = scipy.integrate.quad(moment_1, lowerLim, upperLim)[0] / scipy.integrate.quad(fun, lowerLim, upperLim)[0]
    moment_2 = lambda y: ((y - meanDelay)**2)*fun(y)
    rmsDelay = np.sqrt(scipy.integrate.quad(moment_2, lowerLim, upperLim)[0]/scipy.integrate.quad(fun, lowerLim, upperLim)[0])
    symbolRate = 1/(10*rmsDelay)    # maximum symbol rate to avoid ISI
    coherenceBW = 1/(50*rmsDelay)   # for 0.9 correlation
    # coherenceBW = 1/(5*rmsDelay);%for 0.5 correlation
    return meanDelay, rmsDelay, symbolRate, coherenceBW


fun = lambda tau: np.exp(-tau/0.00001)
meanDelay, rmsDelay, symbolRate, coherenceBW = meas_continuous_PDP(fun, 0, 10e-6)
print(f"meanDelay = {meanDelay}, rmsDelay = {rmsDelay}, symbolRate = {symbolRate}, coherenceBW = {coherenceBW}")


def meas_discrete_PDP(Ps, TAUs):
    # Calculate mean Delay, RMS delay spread and the maximum symbol rate
    # that a signal can be transmitted without ISI and the coherence BW for the discrete PDP. The discrete PDP is specified as a list of
    # power values (Ps) in dBm and corresponding time delays (TAUs)
    p_i = 10**(Ps/10) # convert dBm to linear values
    meanDelay = np.sum(p_i*TAUs)/np.sum(p_i)
    rmsDelay = np.sqrt(np.sum(p_i*(TAUs - meanDelay)**2)/np.sum(p_i))
    symbolRate = 1/(10*rmsDelay)   # Recommended max sym rate to avoid ISI
    coherenceBW = 1/(50*rmsDelay)  # 0.9 correlation
    return meanDelay, rmsDelay, symbolRate, coherenceBW

Ps = np.array([-20, -10, -10, 0])
TAUs = np.array([0, 1e-6, 2e-6, 5e-6])
meanDelay, rmsDelay, symbolRate, coherenceBW = meas_discrete_PDP(Ps, TAUs)
print(f"meanDelay = {meanDelay}, rmsDelay = {rmsDelay}, symbolRate = {symbolRate}, coherenceBW = {coherenceBW}")


#%% Program 11.7: rician pdf.m: Generating Ricean ﬂat-fading samples and plotting its PDF
# Simulate receieved signal samples due to Ricean flat-fading
N = 10**5     # number of received signal samples to generate
K_factors = np.array([0, 3, 7, 12, 20])   # Ricean K factors
colors = ['b','r','k','g','m']
Omega = 1                           # Total average power set to unity

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
x = np.arange(0, 3.01, 0.01)  # Ricean rv
for i, K in enumerate(K_factors):    #  simulate for each K factors
    g1 = np.sqrt(K/((K+1)))
    g2 =  np.sqrt(1/((K+1)))
    # Generate 10^5 Rice fading samples with unity average power
    # r = (g2 * np.random.randn(N) + g1) + 1j * (g2 * np.random.randn(N) + g1)
    r = g1 * (1 + 1j) / np.sqrt(2) + g2 * (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

    z = 2 * x * np.sqrt(K*(K+1)/Omega)   # to generate modified Bessel function
    I0_z = scipy.special.iv(0, z)        # modified Bessel function of first kind, use iv not jv.
    pdf = (2 * x * (K+1) / Omega) * np.exp(-K - (K+1) * x**2 / Omega) * I0_z

    axs.hist(np.abs(r), 100, density = 1, histtype = 'step', color = colors[i], lw = 1, label = f"K = {K}, Simulation")
    axs.plot(x, pdf,  color = colors[i], lw = 1, ls = 'none', marker = 'o', markevery = 20, label = f"K = {K}, Theory")

axs.legend(labelspacing = 0.01)
axs.set_xlabel( 'x',)
axs.set_ylabel(r'$f_{\xi}$(x)',)
axs.set_title("PDF of envelope of received signal")
plt.show()
plt.close()


#%% Program 11.8: doppler psd acf.m: Generating Ricean ﬂat-fading samples and plotting its PDF
from Tools import freqDomainView
def doppler_psd_acf(TYPE, T_s, f_max, sigma_0_2, N):
    # Jakes or Gaussian PSD and their auto-correlation functions (ACF)
    # TYPE='JAKES' or 'GAUSSIAN'
    # Ts = sampling frequency in seconds
    # f_max = maximum doppler frequency in Hz
    # sigma_0_2 variance of gaussian processes (sigma_0^2)
    # N = number of samples in ACF plot
    # EXAMPLE: doppler_psd_acf('JAKES',0.001,91,1,1024)

    tau = T_s * np.arange(-N/2, N/2 + 1)  # time intervals for ACF computation
    if TYPE == 'Jakes':
        r = sigma_0_2 * scipy.special.jv(0, 2 * np.pi * f_max * tau)  #  JAKES ACF
    elif TYPE == 'Gaussian':
        f_c = np.sqrt(np.log(2)) * f_max   #  3-dB cut-off frequency
        r = sigma_0_2 * np.exp(-(np.pi * f_c / np.sqrt(np.log(2)) * tau)**2)  #  Gaussian ACF
    else:
        print('Invalid PSD TYPE specified')

    # Power spectral density using FFT of ACF
    f, Y, _, _, _, _ = freqDomainView(r, 1/T_s, 'double')  #  chapter 1, section 1.3.4

    ##### plot
    fig, axs = plt.subplots(1, 2, figsize = (12, 5), constrained_layout = True)

    axs[0].plot(f, np.abs(Y), lw = 1, color = 'b',  label = ' ')
    axs[0].set_xlabel('f(Hz)',)
    axs[0].set_ylabel(r'$S_{\mu_i\mu_i}(f)$',)
    axs[0].set_title(f'{TYPE} PSD' )
    # axs[0].set_xlim(-200, 200)

    axs[1].plot(tau, r, color = 'r', lw = 1, label = ' ')
    axs[1].set_xlabel(r'\tau(s)',)
    axs[1].set_ylabel(r'$r_{\mu_i\mu_i}(\tau)$',)
    axs[1].set_title('Auto-correlation fn.')
    # axs[1].set_xlim(-20,20)
    # plt.suptitle(f"Impulse response & spectrum of windowed Jakes filter ( fmax = {fd}Hz, Ts = {Ts}s, N = {N})", fontsize = 22)
    plt.show()
    plt.close()

doppler_psd_acf('Jakes', 0.001, 91, 1, 200)


#%% Program 11.9: param MEDS.m: Generate parameters for deterministic model using MEDS method




#%% Program 11.10: Rice method.m: Function to simulate deterministic Rice model




#%% Program 11.11: pdp model.m: TDL implementation of specified power delay profile




#%% Program 11.12: freq selective TDL model.m: Simulate frequency selective Rayleigh block fading channel



























































































































































































































































