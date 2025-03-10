"""
Passband simulation models - modulation and demodulation techniques (Chapter 2)

@author: Mathuranathan Viswanathan
Created on Jul 17, 2019
"""
import numpy as np
import matplotlib.pyplot as plt

def bpsk_mod(ak,L):
    """
    Function to modulate an incoming binary stream using BPSK (baseband)
    Parameters:
        ak : input binary data stream (0's and 1's) to modulate
        L : oversampling factor (Tb/Ts)
    Returns:
        (s_bb,t) : tuple of following variables
                s_bb: BPSK modulated signal(baseband) - s_bb(t)
                 t :  generated time base for the modulated signal
    """
    from scipy.signal import upfirdn
    s_bb = upfirdn(h=[1]*L, x=2*ak-1, up = L) # NRZ encoder
    t=np.arange(start = 0,stop = len(ak)*L) #discrete time base
    return (s_bb,t)

def bpsk_demod(r_bb,L):
    """
    Function to demodulate a BPSK (baseband) signal
    Parameters:
        r_bb : received signal at the receiver front end (baseband)
        L : oversampling factor (Tsym/Ts)
    Returns:
        ak_hat : detected/estimated binary stream
    """
    x = np.real(r_bb) # I arm
    x = np.convolve(x,np.ones(L)) # integrate for Tb duration (L samples)
    x = x[L-1:-1:L] # I arm - sample at every L
    ak_hat = (x > 0).transpose() # threshold detector
    return ak_hat

def add_awgn_noise(s,SNRdB,L=1):
    """
    Function to add AWGN noise to input signal
    The function adds AWGN noise vector to signal 's' to generate a \
    resulting signal vector 'r' of specified SNR in dB. It also returns\
    the noise vector 'n' that is added to the signal 's' and the spectral\
    density N0 of noise added
    
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB)
            for the received signal
        L : oversampling factor (applicable for waveform simulation)
            default L = 1.
    Returns:
        (r,n,N0) : tuple of following variables
                r : received signal vector (r = s+n)
                n : noise signal vector added
                N0 : spectral density of the generated noise vector n
    """
    gamma = 10**(SNRdB/10) #SNR to linear scale
    
    if s.ndim==1:# if s is single dimensional vector
        P=L*np.sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*np.sum(np.sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
        
    N0=P/gamma # Find the noise spectral density
    
    from numpy.random import standard_normal
    if np.isrealobj(s):# check if input is real/complex object type
        n = np.sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = np.sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape)) #computed noise
    r = s + n # received signal    
    return (r,n,N0)

def qpsk_mod(a, fc, OF, enable_plot = False):
    """
    Modulate an incoming binary stream using conventional QPSK
    Parameters:
        a : input binary data stream (0's and 1's) to modulate
        fc : carrier frequency in Hertz
        OF : oversampling factor - at least 4 is better
        enable_plot : True = plot transmitter waveforms (default False)
    Returns:
        result : Dictionary containing the following keyword entries:
          s(t) : QPSK modulated signal vector with carrier i.e, s(t)
          I(t) : baseband I channel waveform (no carrier)
          Q(t) : baseband Q channel waveform (no carrier)
          t : time base for the carrier modulated signal
    """
    L = 2*OF # samples in each symbol (QPSK has 2 bits in each symbol)
    I = a[0::2];Q = a[1::2] #even and odd bit streams
    # even/odd streams at 1/2Tb baud
    from scipy.signal import upfirdn #NRZ encoder
    I = upfirdn(h=[1]*L, x=2*I-1, up = L)
    Q = upfirdn(h=[1]*L, x=2*Q-1, up = L)
    
    fs = OF*fc # sampling frequency 
    t=np.arange(0,len(I)/fs,1/fs)  #time base
    I_t = I*np.cos(2*np.pi*fc*t);Q_t = -Q*np.sin(2*np.pi*fc*t)
    s_t = I_t + Q_t # QPSK modulated baseband signal
    
    if enable_plot:
        fig = plt.figure(constrained_layout=True)
        
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0]);ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0]);ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[-1,:])
        
        # show first few symbols of I(t), Q(t)
        ax1.plot(t,I);ax1.set_title('I(t)')        
        ax2.plot(t,Q);ax2.set_title('Q(t)')
        ax3.plot(t,I_t,'r');ax3.set_title('$I(t) cos(2 \pi f_c t)$')
        ax4.plot(t,Q_t,'r');ax4.set_title('$Q(t) sin(2 \pi f_c t)$')
        ax1.set_xlim(0,20*L/fs);ax2.set_xlim(0,20*L/fs)
        ax3.set_xlim(0,20*L/fs);ax4.set_xlim(0,20*L/fs)
        ax5.plot(t,s_t);ax5.set_xlim(0,20*L/fs);fig.show()
        ax5.set_title('$s(t) = I(t) cos(2 \pi f_c t) - Q(t) sin(2 \pi f_c t)$')
    result = dict()
    result['s(t)'] =s_t;result['I(t)'] = I;result['Q(t)'] = Q;result['t'] = t           
    return result
        
def qpsk_demod(r,fc,OF,enable_plot=False):
    """
    Demodulate a conventional QPSK signal
    Parameters:
        r : received signal at the receiver front end
        fc : carrier frequency (Hz)
        OF : oversampling factor (at least 4 is better)
        enable_plot : True = plot receiver waveforms (default False)
    Returns:
        a_hat - detected binary stream
    """
    fs = OF*fc # sampling frequency
    L = 2*OF # number of samples in 2Tb duration
    t=np.arange(0,len(r)/fs,1/fs) # time base
    x=r*np.cos(2*np.pi*fc*t) # I arm
    y=-r*np.sin(2*np.pi*fc*t) # Q arm
    x = np.convolve(x,np.ones(L)) # integrate for L (Tsym=2*Tb) duration
    y = np.convolve(y,np.ones(L)) #integrate for L (Tsym=2*Tb) duration
    
    x = x[L-1::L] # I arm - sample at every symbol instant Tsym
    y = y[L-1::L] # Q arm - sample at every symbol instant Tsym
    a_hat = np.zeros(2*len(x))
    a_hat[0::2] = (x>0) # even bits
    a_hat[1::2] = (y>0) # odd bits
    
    if enable_plot:
        fig, axs = plt.subplots(1, 1)
        axs.plot(x[0:200],y[0:200],'o');fig.show()
                
    return a_hat

def oqpsk_mod(a,fc,OF,enable_plot=False):
    """
    Modulate an incoming binary stream using OQPSK
    Parameters:
        a : input binary data stream (0's and 1's) to modulate
        fc : carrier frequency in Hertz
        OF : oversampling factor - at least 4 is better
        enable_plot : True = plot transmitter waveforms (default False)
    Returns:
        result : Dictionary containing the following keyword entries:
          s(t) : QPSK modulated signal vector with carrier i.e, s(t)
          I(t) : baseband I channel waveform (no carrier)
          Q(t) : baseband Q channel waveform (no carrier)
          t : time base for the carrier modulated signal
    """
    L = 2*OF # samples in each symbol (QPSK has 2 bits in each symbol)
    I = a[0::2];Q = a[1::2] #even and odd bit streams
    # even/odd streams at 1/2Tb baud
    from scipy.signal import upfirdn #NRZ encoder
    I = upfirdn(h=[1]*L, x=2*I-1, up = L)
    Q = upfirdn(h=[1]*L, x=2*Q-1, up = L)

    I = np.hstack((I,np.zeros(L//2))) # padding at end
    Q = np.hstack((np.zeros(L//2),Q)) # padding at start
    
    fs = OF*fc # sampling frequency 
    t=np.arange(0,len(I)/fs,1/fs)  #time base
    I_t = I*np.cos(2*np.pi*fc*t);Q_t = -Q*np.sin(2*np.pi*fc*t)
    s = I_t + Q_t # QPSK modulated baseband signal
    
    if enable_plot:
        fig = plt.figure(constrained_layout=True)
        
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0]);ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0]);ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[-1,:])
        
        # show first few symbols of I(t), Q(t)
        ax1.plot(t,I);ax1.set_title('I(t)')        
        ax2.plot(t,Q);ax2.set_title('Q(t)')
        ax3.plot(t,I_t,'r');ax3.set_title('$I(t) cos(2 \pi f_c t)$')
        ax4.plot(t,Q_t,'r');ax4.set_title('$Q(t) sin(2 \pi f_c t)$')
        ax1.set_xlim(0,20*L/fs);ax2.set_xlim(0,20*L/fs)
        ax3.set_xlim(0,20*L/fs);ax4.set_xlim(0,20*L/fs)
        ax5.plot(t,s);ax5.set_xlim(0,20*L/fs);fig.show()
        ax5.set_title('$s(t) = I(t) cos(2 \pi f_c t) - Q(t) sin(2 \pi f_c t)$')
        
        fig, axs = plt.subplots(1, 1)
        axs.plot(I,Q);fig.show()#constellation plot
    result = dict()
    result['s(t)'] =s;result['I(t)'] = I;result['Q(t)'] = Q;result['t'] = t           
    return result

def oqpsk_demod(r,N,fc,OF,enable_plot=False):
    """
    Demodulate a OQPSK signal
    Parameters:
        r : received signal at the receiver front end
        N : Number of OQPSK symbols transmitted
        fc : carrier frequency (Hz)
        OF : oversampling factor (at least 4 is better)
        enable_plot : True = plot receiver waveforms (default False)
    Returns:
        a_hat - detected binary stream
    """
    fs = OF*fc # sampling frequency
    L = 2*OF # number of samples in 2Tb duration
    t=np.arange(0,(N+1)*OF/fs,1/fs) # time base
    x=r*np.cos(2*np.pi*fc*t) # I arm
    y=-r*np.sin(2*np.pi*fc*t) # Q arm
    x = np.convolve(x,np.ones(L)) # integrate for L (Tsym=2*Tb) duration
    y = np.convolve(y,np.ones(L)) #integrate for L (Tsym=2*Tb) duration
    
    x = x[L-1:-1-L:L] # I arm - sample at every symbol instant Tsym
    y = y[L+L//2-1:-1-L//2:L] # Q arm - sample at every symbol starting at L+L/2-1th sample
    a_hat = np.zeros(N)
    a_hat[0::2] = (x>0) # even bits
    a_hat[1::2] = (y>0) # odd bits
    
    if enable_plot:
        fig, axs = plt.subplots(1, 1)
        axs.plot(x[0:200],y[0:200],'o');fig.show()                
    return a_hat

def piBy4_dqpsk_Diff_encoding(a,enable_plot=False):
    """
    Phase Mapper for pi/4-DQPSK modulation
    Parameters:
        a : input stream of binary bits
    Returns:
        (u,v): tuple, where
           u : differentially coded I-channel bits
           v : differentially coded Q-channel bits    
    """
    from numpy import pi, cos, sin
    if len(a)%2: raise ValueError('Length of binary stream must be even')
    I = a[0::2] # odd bit stream
    Q = a[1::2] # even bit stream
    # club 2-bits to form a symbol and use it as index for dTheta table
    m = 2*I+Q
    dTheta = np.array([-3*pi/4, 3*pi/4, -pi/4, pi/4]) #LUT for pi/4-DQPSK
    u = np.zeros(len(m)+1);v = np.zeros(len(m)+1)
    u[0]=1; v[0]=0 # initial conditions for uk and vk
    for k in range(0,len(m)):
        u[k+1] = u[k] * cos(dTheta[m[k]]) - v[k] * sin(dTheta[m[k]])
        v[k+1] = u[k] * sin(dTheta[m[k]]) + v[k] * cos(dTheta[m[k]])
    if enable_plot:#constellation plot
        fig, axs = plt.subplots(1, 1)
        axs.plot(u,v,'o');
        axs.set_title('Constellation');fig.show()
    return (u,v)

def piBy4_dqpsk_mod(a,fc,OF,enable_plot = False):
    """
    Modulate a binary stream using pi/4 DQPSK
    Parameters:
        a : input binary data stream (0's and 1's) to modulate
        fc : carrier frequency in Hertz
        OF : oversampling factor
    Returns:
        result : Dictionary containing the following keyword entries:
          s(t) : pi/4 QPSK modulated signal vector with carrier
          U(t) : differentially coded I-channel waveform (no carrier)
          V(t) : differentially coded Q-channel waveform (no carrier)
          t: time base
    """
    (u,v)=piBy4_dqpsk_Diff_encoding(a) # Differential Encoding for pi/4 QPSK
    #Waveform formation (similar to conventional QPSK)
    L = 2*OF # number of samples in each symbol (QPSK has 2 bits/symbol)
    
    U = np.tile(u, (L,1)).flatten('F')# odd bit stream at 1/2Tb baud
    V = np.tile(v, (L,1)).flatten('F')# even bit stream at 1/2Tb baud
    
    fs = OF*fc # sampling frequency
    
    t=np.arange(0, len(U)/fs,1/fs) #time base
    U_t = U*np.cos(2*np.pi*fc*t)
    V_t = -V*np.sin(2*np.pi*fc*t)
    s_t = U_t + V_t
    
    if enable_plot:
        fig = plt.figure(constrained_layout=True)
        
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0]);ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0]);ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[-1,:])
        
        ax1.plot(t,U);ax2.plot(t,V)
        ax3.plot(t,U_t,'r');ax4.plot(t,V_t,'r')
        ax5.plot(t,s_t) #QPSK waveform zoomed to first few symbols
        
        ax1.set_ylabel('U(t)-baseband');ax2.set_ylabel('V(t)-baseband')
        ax3.set_ylabel('U(t)-with carrier');ax4.set_ylabel('V(t)-with carrier')
        ax5.set_ylabel('s(t)');ax5.set_xlim([0,10*L/fs])
        ax1.set_xlim([0,10*L/fs]);ax2.set_xlim([0,10*L/fs])
        ax3.set_xlim([0,10*L/fs]);ax4.set_xlim([0,10*L/fs])
        fig.show()
        
    result = dict()
    result['s(t)'] =s_t;result['U(t)'] = U;result['V(t)'] = V;result['t'] = t
    return result

def piBy4_dqpsk_Diff_decoding(w,z):
    """
    Phase Mapper for pi/4-DQPSK modulation
    Parameters:
        w - differentially coded I-channel bits at the receiver
        z - differentially coded Q-channel bits at the receiver
    Returns:
        a_hat - binary bit stream after differential decoding
    """
    if len(w)!=len(z): raise ValueError('Length mismatch between w and z')
    x = np.zeros(len(w)-1);y = np.zeros(len(w)-1);
    
    for k in range(0,len(w)-1):
        x[k] = w[k+1]*w[k] + z[k+1]*z[k]
        y[k] = z[k+1]*w[k] - w[k+1]*z[k]
    
    a_hat = np.zeros(2*len(x))
    a_hat[0::2] = (x > 0) # odd bits
    a_hat[1::2] = (y > 0) # even bits
    return a_hat

def piBy4_dqpsk_demod(r,fc,OF,enable_plot=False):
    """
    Differential coherent demodulation of pi/4-DQPSK
    Parameters:
        r : received signal at the receiver front end
        fc : carrier frequency in Hertz
        OF : oversampling factor (multiples of fc) - at least 4 is better
    Returns:
        a_cap :  detected binary stream
    """
    fs = OF*fc # sampling frequency
    L = 2*OF # samples in 2Tb duration
    t=np.arange(0, len(r)/fs,1/fs)
    w=r*np.cos(2*np.pi*fc*t) # I arm
    z=-r*np.sin(2*np.pi*fc*t) # Q arm
    w = np.convolve(w,np.ones(L)) # integrate for L (Tsym=2*Tb) duration
    z = np.convolve(z,np.ones(L)) # integrate for L (Tsym=2*Tb) duration
    w = w[L-1::L] # I arm - sample at every symbol instant Tsym
    z = z[L-1::L] # Q arm - sample at every symbol instant Tsym
    a_cap = piBy4_dqpsk_Diff_decoding(w,z)
    
    if enable_plot:#constellation plot
        fig, axs = plt.subplots(1, 1)
        axs.plot(w,z,'o')
        axs.set_title('Constellation');fig.show()
    return a_cap

def msk_mod(a, fc, OF, enable_plot = False):
    """
    Modulate an incoming binary stream using MSK
    Parameters:
        a : input binary data stream (0's and 1's) to modulate
        fc : carrier frequency in Hertz
        OF : oversampling factor (at least 4 is better)
    Returns:
        result : Dictionary containing the following keyword entries:
          s(t) : MSK modulated signal with carrier
          sI(t) : baseband I channel waveform(no carrier)
          sQ(t) : baseband Q channel waveform(no carrier)
          t: time base
    """ 
    ak = 2*a-1 # NRZ encoding 0-> -1, 1->+1
    ai = ak[0::2]; aq = ak[1::2] # split even and odd bit streams
    L = 2*OF # represents one symbol duration Tsym=2xTb
    
    #upsample by L the bits streams in I and Q arms
    from scipy.signal import upfirdn, lfilter
    ai = upfirdn(h=[1], x=ai, up = L)
    aq = upfirdn(h=[1], x=aq, up = L)
    
    aq = np.pad(aq, (L//2,0), 'constant') # delay aq by Tb (delay by L/2)
    ai = np.pad(ai, (0,L//2), 'constant') # padding at end to equal length of Q
    
    #construct Low-pass filter and filter the I/Q samples through it
    Fs = OF*fc;Ts = 1/Fs;Tb = OF*Ts
    t = np.arange(0,2*Tb+Ts,Ts)
    h = np.sin(np.pi*t/(2*Tb))# LPF filter
    sI_t = lfilter(b = h, a = [1], x = ai) # baseband I-channel
    sQ_t = lfilter(b = h, a = [1], x = aq) # baseband Q-channel
    
    t=np.arange(0, Ts*len(sI_t), Ts) # for RF carrier
    sIc_t = sI_t*np.cos(2*np.pi*fc*t) #with carrier
    sQc_t = sQ_t*np.sin(2*np.pi*fc*t) #with carrier
    s_t =  sIc_t - sQc_t# Bandpass MSK modulated signal
    
    if enable_plot:
        fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
        
        ax1.plot(t,sI_t);ax1.plot(t,sIc_t,'r')
        ax2.plot(t,sQ_t);ax2.plot(t,sQc_t,'r')
        ax3.plot(t,s_t,'--')        
        ax1.set_ylabel('$s_I(t)$');ax2.set_ylabel('$s_Q(t)$')
        ax3.set_ylabel('s(t)')        
        ax1.set_xlim([-Tb,20*Tb]);ax2.set_xlim([-Tb,20*Tb])
        ax3.set_xlim([-Tb,20*Tb])
        fig.show()
        
    result = dict()
    result['s(t)'] =s_t;result['sI(t)'] = sI_t;result['sQ(t)'] = sQ_t;result['t'] = t
    return result

def msk_demod(r,N,fc,OF):
    """
    MSK demodulator
    Parameters:
        r : received signal at the receiver front end
        N : number of symbols transmitted
        fc : carrier frequency in Hertz
        OF : oversampling factor (at least 4 is better)
    Returns:
        a_hat : detected binary stream
    """
    L = 2*OF # samples in 2Tb duration
    Fs=OF*fc;Ts=1/Fs;Tb = OF*Ts; # sampling frequency, durations
    t=np.arange(-OF, len(r) - OF)/Fs # time base
    
    # cosine and sine functions for half-sinusoid shaping
    x=abs(np.cos(np.pi*t/(2*Tb)));y=abs(np.sin(np.pi*t/(2*Tb)))
    
    u=r*x*np.cos(2*np.pi*fc*t) # multiply I by half cosines and cos(2pifct)
    v=-r*y*np.sin(2*np.pi*fc*t) # multiply Q by half sines and sin(2pifct)
    
    iHat = np.convolve(u,np.ones(L)) # integrate for L (Tsym=2*Tb) duration
    qHat = np.convolve(v,np.ones(L)) # integrate for L (Tsym=2*Tb) duration
    
    iHat= iHat[L-1:-1-L:L]  # I- sample at the end of every symbol
    qHat= qHat[L+L//2-1:-1-L//2:L] # Q-sample from L+L/2th sample
    
    a_hat = np.zeros(N)
    a_hat[0::2] = iHat > 0 # thresholding - odd bits
    a_hat[1::2] = qHat > 0 # thresholding - even bits
    
    return a_hat

def gaussianLPF(BT, Tb, L, k):
    """
    Function to generate filter coefficients of Gaussian low pass filter (used in gmsk_mod)
    Parameters:
        BT : BT product - Bandwidth x bit period
        Tb : bit period
        L : oversampling factor (number of samples per bit)
        k : span length of the pulse (bit interval)        
    Returns:
        h_norm : normalized filter coefficients of Gaussian LPF
    """
    B = BT/Tb # bandwidth of the filter
    t = np.arange(start = -k*Tb, stop = k*Tb + Tb/L, step = Tb/L) # truncated time limits for the filter
    h = B*np.sqrt(2*np.pi/(np.log(2)))*np.exp(-2 * (t*np.pi*B)**2 /(np.log(2)))
    h_norm=h/np.sum(h)
    return h_norm

def gmsk_mod(a,fc,L,BT,enable_plot=False):
    """
    Function to modulate a binary stream using GMSK modulation
    Parameters:
        BT : BT product (bandwidth x bit period) for GMSK
        a : input binary data stream (0's and 1's) to modulate
        fc : RF carrier frequency in Hertz
        L : oversampling factor
        enable_plot: True = plot transmitter waveforms (default False)
    Returns:
        (s_t,s_complex) : tuple containing the following variables
            s_t : GMSK modulated signal with carrier s(t)
            s_complex : baseband GMSK signal (I+jQ)
    """
    from scipy.signal import upfirdn,lfilter
    
    fs = L*fc; Ts=1/fs;Tb = L*Ts; # derived waveform timing parameters
    c_t = upfirdn(h=[1]*L, x=2*a-1, up = L) #NRZ pulse train c(t)
    
    k=1 # truncation length for Gaussian LPF
    h_t = gaussianLPF(BT,Tb,L,k) # Gaussian LPF with BT=0.25
    b_t = np.convolve(h_t,c_t,'full') # convolve c(t) with Gaussian LPF to get b(t)
    bnorm_t = b_t/max(abs(b_t)) # normalize the output of Gaussian LPF to +/-1
    
    h = 0.5;
    phi_t = lfilter(b = [1], a = [1,-1], x = bnorm_t*Ts) * h*np.pi/Tb # integrate to get phase information
    
    I = np.cos(phi_t)
    Q = np.sin(phi_t) # cross-correlated baseband I/Q signals
    
    s_complex = I - 1j*Q # complex baseband representation
    t = Ts* np.arange(start = 0, stop = len(I)) # time base for RF carrier
    sI_t = I*np.cos(2*np.pi*fc*t); sQ_t = Q*np.sin(2*np.pi*fc*t)
    s_t = sI_t - sQ_t # s(t) - GMSK with RF carrier
    
    if enable_plot:
        fig, axs = plt.subplots(2, 4)
        axs[0,0].plot(np.arange(0,len(c_t))*Ts,c_t);axs[0,0].set_title('c(t)');axs[0,0].set_xlim(0,40*Tb)
        axs[0,1].plot(np.arange(-k*Tb,k*Tb+Ts,Ts),h_t);axs[0,1].set_title('$h(t): BT_b$='+str(BT))
        axs[0,2].plot(t,I,'--');axs[0,2].plot(t,sI_t,'r');axs[0,2].set_title('$I(t)cos(2 \pi f_c t)$');axs[0,2].set_xlim(0,10*Tb)
        axs[0,3].plot(t,Q,'--');axs[0,3].plot(t,sQ_t,'r');axs[0,3].set_title('$Q(t)sin(2 \pi f_c t)$');axs[0,3].set_xlim(0,10*Tb)
        axs[1,0].plot( np.arange(0,len(bnorm_t))*Ts,bnorm_t);axs[1,0].set_title('b(t)');axs[1,0].set_xlim(0,40*Tb)
        axs[1,1].plot(np.arange(0,len(phi_t))*Ts, phi_t);axs[1,1].set_title('$\phi(t)$')
        axs[1,2].plot(t,s_t);axs[1,2].set_title('s(t)');axs[1,2].set_xlim(0,20*Tb)
        axs[1,3].plot(I,Q);axs[1,3].set_title('constellation')
        fig.show()
        
    return (s_t,s_complex)
def gmsk_demod(r_complex,L):
    """
    Function to demodulate a baseband GMSK signal
    Parameters:
        r_complex : received signal at receiver front end (complex form - I+jQ)
        L : oversampling factor
    Returns:
        a_hat : detected binary stream
    """
    I=np.real(r_complex); Q = -np.imag(r_complex); # I,Q streams
    z1 = Q * np.hstack((np.zeros(L), I[0:len(I)-L]))
    z2 = I * np.hstack((np.zeros(L), Q[0:len(I)-L]))
    z = z1 - z2
    a_hat = (z[2*L-1:-L:L] > 0).astype(int) # sampling and hard decision
    #sampling indices depends on the truncation length (k) of Gaussian LPF defined in the modulator
    return a_hat

def bfsk_mod(a,fc,fd,L,fs,fsk_type='coherent',enable_plot = False):
    """
    Function to modulate an incoming binary stream using BFSK
    Parameters:
        a : input binary data stream (0's and 1's) to modulate
        fc : center frequency of the carrier in Hertz
        fd : frequency separation measured from Fc
        L : number of samples in 1-bit period
        fs : Sampling frequency for discrete-time simulation
        fsk_type : 'coherent' (default) or 'noncoherent' FSK generation
        enable_plot: True = plot transmitter waveforms (default False)
    Returns:
        (s_t,phase) : tuple containing following parameters
            s_t : BFSK modulated signal
            phase : initial phase generated by modulator, applicable only for coherent FSK. It can be used when using coherent detection at Rx    
    """
    from scipy.signal import upfirdn
    a_t = upfirdn(h=[1]*L, x=a, up = L) #data to waveform
    
    t = np.arange(start=0,stop=len(a_t))/fs #time base    
    if fsk_type.lower() == 'noncoherent':
        c1 = np.cos(2*np.pi*(fc+fd/2)*t+2*np.pi*np.random.random_sample()) # carrier 1 with random phase
        c2 = np.cos(2*np.pi*(fc-fd/2)*t+2*np.pi*np.random.random_sample()) # carrier 2 with random phase
    else: #coherent is default
        phase=2*np.pi*np.random.random_sample() # random phase from uniform distribution [0,2pi)
        c1 = np.cos(2*np.pi*(fc+fd/2)*t+phase) # carrier 1 with random phase
        c2 = np.cos(2*np.pi*(fc-fd/2)*t+phase) # carrier 2 with the same random phase
    s_t = a_t*c1 +(-a_t+1)*c2 # BFSK signal (MUX selection)
    
    if enable_plot:
        fig, (ax1,ax2) = plt.subplots(2, 1)
        ax1.plot(t,a_t);ax2.plot(t,s_t);fig.show()
        
    return (s_t,phase)

def bfsk_coherent_demod(r_t,phase,fc,fd,L,fs):
    """
    Coherent demodulation of BFSK modulated signal
    Parameters:
        r_t : BFSK modulated signal at the receiver r(t)
        phase : initial phase generated at the transmitter
        fc : center frequency of the carrier in Hertz
        fd : frequency separation measured from Fc
        L : number of samples in 1-bit period
        fs : Sampling frequency for discrete-time simulation    
    Returns:
        a_hat : data bits after demodulation        
    """
    t = np.arange(start=0,stop=len(r_t))/fs # time base
    x = r_t*(np.cos(2*np.pi*(fc+fd/2)*t+phase)-np.cos(2*np.pi*(fc-fd/2)*t+phase))
    y = np.convolve(x,np.ones(L)) # integrate/sum from 0 to L
    a_hat = (y[L-1::L]>0).astype(int) # sample at every sampling instant and detect
    return a_hat

def bfsk_noncoherent_demod(r_t,fc,fd,L,fs):
    """
    Non-coherent demodulation of BFSK modulated signal
    Parameters:
        r_t : BFSK modulated signal at the receiver r(t)
        fc : center frequency of the carrier in Hertz
        fd : frequency separation measured from Fc
        L : number of samples in 1-bit period
        fs : Sampling frequency for discrete-time simulation
    Returns:
        a_hat : data bits after demodulation        
    """
    t = np.arange(start=0,stop=len(r_t))/fs # time base
    f1 = (fc+fd/2); f2 = (fc-fd/2)
    #define four basis functions
    p1c = np.cos(2*np.pi*f1*t); p2c = np.cos(2*np.pi*f2*t)
    p1s = -1*np.sin(2*np.pi*f1*t); p2s = -1*np.sin(2*np.pi*f2*t)
    # multiply and integrate from 0 to L
    r1c = np.convolve(r_t*p1c,np.ones(L)); r2c = np.convolve(r_t*p2c,np.ones(L))
    r1s = np.convolve(r_t*p1s,np.ones(L)); r2s = np.convolve(r_t*p2s,np.ones(L))
    # sample at every sampling instant
    r1c = r1c[L-1::L]; r2c = r2c[L-1::L]
    r1s = r1s[L-1::L]; r2s = r2s[L-1::L]
    # square and add
    x = r1c**2 + r1s**2
    y = r2c**2 + r2s**2
    a_hat=((x-y)>0).astype(int) # compare and decide
    return a_hat