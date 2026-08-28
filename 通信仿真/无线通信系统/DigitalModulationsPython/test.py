

##>>>>>>>>>>>>>>>>>>>>>>>>   OFDM 理论上的等效过程，

import numpy as np

def cconv(a, b, n=None):
    """
    实现与MATLAB cconv完全一致的圆卷积
    参数:
        a, b: 输入复数数组
        n: 输出长度 (None表示默认长度len(a)+len(b)-1)
    返回:
        圆卷积结果 (复数数组)
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    # 默认输出长度
    if n is None:
        n = len(a) + len(b) - 1
    # 线性卷积
    linear_conv = np.convolve(a, b, mode='full')
    # 处理不同n的情况
    if n <= 0:
        return np.array([], dtype=complex)
    result = np.zeros(n, dtype=complex)
    if n <= len(linear_conv):
        # n <= M+N-1: 重叠相加
        for k in range(n):
            # 收集所有k + m*n位置的元素
            idx = np.arange(k, len(linear_conv), n)
            result[k] = np.sum(linear_conv[idx])
    else:
        # n > M+N-1: 补零
        result[:len(linear_conv)] = linear_conv

    return result

# 左下角N-k, 维度N
def Jap_Nk_right(N, k):
    Jtilde = np.zeros((N, N))
    Jtilde[k:N, 0:N - k] = np.eye(N - k)
    return Jtilde

# 右上角N-k, 维度N
def Jap_Nk_left(N, k):
    Jtilde = np.zeros((N, N))
    Jtilde[0:N - k, k:N] = np.eye(N - k)
    return Jtilde
# 左下角N-k，右上角k, 维度N
def Jp_Nk_right(N, k):
    Jtilde = np.zeros((N, N))
    Jtilde[k:N, 0:N - k] = np.eye(N - k)
    Jtilde[0:k, N - k:N] = np.eye(k)
    return Jtilde
# 左下角 k，右上角 N-k, 维度N
def Jp_Nk_left(N, k):
    Jtilde = np.zeros((N, N))
    Jtilde[0:N - k, k:N] = np.eye(N - k)
    Jtilde[N - k:N, 0:k] = np.eye(k)
    return Jtilde


# 产生傅里叶矩阵
def FFTmatrix(L, ):
     mat = np.zeros((L, L), dtype = complex)
     ll = np.arange(L)
     for i in range(L):
         mat[i,:] = 1.0*np.exp(-1j*2.0*np.pi*i*ll/L) / (np.sqrt(L)*1.0)
     return mat


def AcpMat(N, Ncp):
    Acp = np.block([
        [np.zeros((Ncp, N-Ncp)), np.eye(Ncp)],
        [np.eye(N)]
        ])
    return Acp

def ScpMat(N, Ncp, Lh):
    Scp = np.block([
        [np.zeros((N, Ncp)), np.eye(N), np.zeros((N, Lh-1))]
        ])
    return Scp

def RcpMat(N, Ncp):
    Rcp = np.block([
        [np.zeros((N, Ncp)), np.eye(N)]
    ])
    return Rcp
def convMatrix(h, N):  #
    """
    Construct the convolution matrix of size (L+N-1)x N from the
    input matrix h of size L. (see chapter 1)
    Parameters:
        h : numpy vector of length L
        N : scalar value
    Returns:
        H : convolution matrix of size (L+N-1)xN
    """
    col = np.hstack((h, np.zeros(N-1)))
    row = np.hstack((h[0], np.zeros(N-1)))

    from scipy.linalg import toeplitz
    H = toeplitz(col, row)
    return H

# 下面是OFDM中IFFT -> +cp -> H -> -cp -> FFT的等效过程
h = np.array([-0.4878, -1.5351, 0.2355])
S = np.array([-0.0155, 2.5770, 1.9238, -0.0629])

N = S.size
L = h.size
F = FFTmatrix(N)
FH = F.conj().T

U = FH
s = U @ S                     # IFFT

cir_s_h = cconv(h, s, N)      # circular conv

# Hlin = convMatrix(h, N)
# y = Hlin @ s                # linear conv

Ncp = L - 1
Acp = AcpMat(N, Ncp)
s_cp = Acp @ s                # add CP

Hlin_cp = convMatrix(h, s_cp.size)
y_lincp = Hlin_cp @ s_cp                # pass freq selected channel
y_remo_cp = y_lincp[Ncp:Ncp + N]        # receiver, remove cp, == cir_s_h

Scp = ScpMat(N, Ncp, L)
y_remo_cp1 = Scp @ Hlin_cp @ Acp @ U @ S    #  pass freq selected channel + remove cp

Diag = F @ Scp @ Hlin_cp @ Acp @ U      # Eq.(3): F@T(h)@A@FH is diagonal such that the data is parallelly transmitted over different subcarriers, and thus the ISI is avoided.

Heff = Scp @ Hlin_cp @ Acp
print(f"h = {h}\n Heff = \n{Heff}")     # H --> Hcir, 将拓普利兹矩阵变为循环阵, 到这里，从离散信号角度完美的对应OFDM的理论


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
# colors = ['b', 'g', 'r', 'c', 'm', 'k']
colors = plt.cm.jet(np.linspace(0, 1, len(arrayOfM))) # colormap
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

    axs.semilogy(EbN0dBs, SER_sim, color = colors[m], ls = 'none', marker = "o", ms = 12, )
    axs.semilogy(EbN0dBs, SER_theory, color = colors[m], ls = '-', label = f'{M}-{MOD_TYPE.upper()}' )

axs.set_ylim(1e-3, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"M{MOD_TYPE.upper()}-CP-OFDM over Freq Selective Rayleigh")
axs.legend(fontsize = 20)
out_fig = plt.gcf()
# out_fig.savefig('hh1.png',format='png',dpi=1000,)
plt.show()
plt.close()



#%%##>>>>>>>>>>>>>>>>>>>>>>>>  奈奎斯特传输， 蒙特卡洛仿真,  Performance of modulations in AWGN
# 不使用upfirdn函数，手动实现
#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start = -4, stop = 26, step = 2) # Eb/N0 range in dB for simulation
mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate, [2, 4, 8, 16, 32]
coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

mod_type = 'QAM'
arrayOfM = [4, 16, 64, 256] # uncomment this line if MOD_TYPE='QAM', [4, 16, 64, 256]

beta = 0.3
span = 8
L = 4
# p, t, filtDelay = srrcFunction(beta, L, span)

modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem,'fsk':FSKModem}
colors = plt.cm.jet(np.linspace(0, 1, len(arrayOfM))) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6), constrained_layout = True)

for i, M in enumerate(arrayOfM):
    print(f" {M} in {arrayOfM}")
    #-----Initialization of various parameters----
    k = np.log2(M)
    EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation
    SER_sim = np.zeros(len(EbN0dBs)) # simulated Symbol error rates

    if mod_type.lower()=='fsk':
        modem=modem_dict[mod_type.lower()](M, coherence)  # choose modem from dictionary
    else:                                                 # for all other modulations
        modem = modem_dict[mod_type.lower()](M)           # choose modem from dictionary

    for j, EsN0dB in enumerate(EsN0dBs):

        d = np.random.randint(low=0, high = M, size = nSym) # uniform random symbols from 0 to M-1
        u = modem.modulate(d) #modulate
        ## Upper sample
        v = np.vstack((u, np.zeros((L-1, u.size))))
        v = v.T.flatten()
        ## plus shaping
        s = scipy.signal.convolve(v, p, 'full')

        ## channel
        r = awgn(s, EsN0dB, L)

        ## receiver
        ## match filter
        vCap = scipy.signal.convolve(r, p, 'full')
        ## Down sampling
        u_hat = vCap[int(2 * filtDelay) : int(vCap.size - 2*filtDelay) : L ] #/ L   ## Note: 当p归一化时，这里千万别用/L，当p没有归一化时，这里需要/L

        if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
            dCap = modem.demodulate(u_hat, coherence)
        else: #demodulate (Refer Chapter 3)
            dCap = modem.demodulate(u_hat)

        SER_sim[j] = np.sum(dCap != d)/nSym

    SER_theory = ser_awgn(EbN0dBs, mod_type, M, coherence) #theory SER
    ax.semilogy(EbN0dBs, SER_sim, color = colors[i], marker='o', linestyle='', label='Sim '+str(M)+'-'+mod_type.upper())
    ax.semilogy(EbN0dBs, SER_theory, color = colors[i], linestyle='-', label='Theory '+str(M)+'-'+mod_type.upper())

ax.set_ylim(1e-6, 1)
ax.set_xlabel('Eb/N0(dB)')
ax.set_ylabel('SER ($P_s$)')
ax.set_title('Symbol Error Rate for M-'+str(mod_type)+' over AWGN')
ax.legend(fontsize = 12)
out_fig = plt.gcf()
out_fig.savefig('hh1.png',format='png',dpi=1000,)
plt.show()
plt.close()
