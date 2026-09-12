

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


from tqdm import tqdm
from DigiCommPy.errorRates import ser_rayleigh
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn
#%% Program 14.2: add cyclic prefix.m: Function to add cyclic prefix of length Ncp symbols
def add_cyclic_prefix(x, Ncp):
    s = np.hstack((x[-Ncp:], x))
    return s

#%% Program 14.3: remove cyclic prefix.m: Function to remove the cyclic prefix from the OFDM symbol
def remove_cyclic_prefix(r, Ncp, N):
    y = r[Ncp : Ncp+N]
    return y

#%% Program 14.4: ofdm on awgn.m: OFDM transmission and reception on AWGN channel


#%%##>>>>>>>>>>>>>>>>>>>>>>>>  OFDM 蒙特卡洛仿真

L = 10              ## Number of taps for the frequency selective channel model

nSym = 10000
EbN0dBs = np.arange(-2, 26, 2)
MOD_TYPE = "psk"    ## "pam" "psk",   "fsk" is not suitable.
arrayOfM = [2, 4, 8, 16, 32]

MOD_TYPE = "qam"
arrayOfM = [4, 16, 64, 256]

coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK
modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem,'fsk':FSKModem}

N = 64
Ncp = 16
colors = ['#F65314', '#00A1F1', '#77AC30', '#8A2BE2', '#00A8BB', 'k']
markers = ['s','v','d', 'o', '*', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',]
# colors = plt.cm.hsv(np.linspace(0, 1, len(arrayOfM))) # colormap
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
for m, M in enumerate(arrayOfM):
    print(f"{m}/{len(arrayOfM)}")
    k = int(np.log2(M))
    EsN0dBs = 10*np.log10(k*N/(N + Ncp)) + EbN0dBs
    errors= np.zeros(EsN0dBs.size)

    if MOD_TYPE.lower() == 'fsk':
        modem = modem_dict[MOD_TYPE.lower()](M, coherence)#choose modem from dictionary
    else: # for all other modulations
        modem = modem_dict[MOD_TYPE.lower()](M)#choose modem from dictionary

    for i, EsN0dB in tqdm(enumerate(EsN0dBs)):
        for j, sym in enumerate(range(nSym)):
            ## Transmitter
            d = np.random.randint(low = 0, high = M, size = N)
            X = modem.modulate(d)

            x = scipy.fft.ifft(X, N)
            s = add_cyclic_prefix(x, Ncp)

            ## Channel
            h = (np.random.randn(L) + 1j * np.random.randn(L))/np.sqrt(2)
            H = scipy.fft.fft(h, N)
            hs = scipy.signal.convolve(h, s)
            r = awgn(hs, EsN0dB)

            ## Receiver
            y = remove_cyclic_prefix(r, Ncp, N)
            Y = scipy.fft.fft(y, N)
            V = Y/H  # 信道均衡（直接除以理想信道，这里没有进行信道估计！）
            if MOD_TYPE.lower()=='fsk': #demodulate (Refer Chapter 3)
                dCap = modem.demodulate(V, coherence)
            else: #demodulate (Refer Chapter 3)
                dCap = modem.demodulate(V)

            ## Error Counter
            numErrors = np.sum(d != dCap)
            errors[i] += numErrors
    SER_sim = errors/(nSym * N)
    SER_theory = ser_rayleigh(EbN0dBs, MOD_TYPE, M)

    axs.semilogy(EbN0dBs, SER_theory, color = colors[m], ls = '-', label = f'{M}-{MOD_TYPE.upper()}, Theor' )
    axs.semilogy(EbN0dBs, SER_sim, color = colors[m], ls = 'none', marker = markers[m], ms = 12, mfc = 'none' , mew = 2, label = f'Simul')


axs.grid(linestyle=(0, (5, 10)), linewidth=0.5, )
axs.set_ylim(1e-3, 1)
axs.set_xlabel( r'$E_b/N_0$(dB)',)
axs.set_ylabel('SER',)

font1 = FontProperties(family='Times New Roman', style='normal', size=16)
legend1 = axs.legend(loc='lower left', borderaxespad=0, edgecolor='black', labelspacing=0, prop=font1, ncols=2)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')

bw = 2
axs.spines['bottom'].set_linewidth(bw)
axs.spines['left'].set_linewidth(bw)
axs.spines['right'].set_linewidth(bw)
axs.spines['top'].set_linewidth(bw)

axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)
labels = axs.get_xticklabels()+axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]

# axs.set_title(f"M{MOD_TYPE.upper()}-CP-OFDM over Freq Selective Rayleigh")
axs.legend(fontsize = 20)
out_fig = plt.gcf()
out_fig.savefig('/home/jack/文档/ShareFileSysu/我的论文/ISAC_Nyquist/Figures/QAM_Rayleigh.pdf', )
plt.show()
plt.close()



