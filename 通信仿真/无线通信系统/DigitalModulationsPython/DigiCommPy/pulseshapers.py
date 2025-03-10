"""
Raised cosine pulse shaping filter

@author: Mathuranathan Viswanathan
Created on Jul 26, 2019
"""
import numpy as np

def raisedCosineDesign(alpha, span, L):
    """
    Raised cosine FIR filter design
    Parameters:
        alpha : roll-off factor
        span : filter span in symbols 
        L : oversampling factor (i.e, each symbol contains L samples)
    Returns:
        p - filter coefficients b of the designed
            FIR raised cosine filter
    """
    t = np.arange(-span/2, span/2 + 1/L, 1/L) # +/- discrete-time base
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.divide(np.sin(np.pi*t),(np.pi*t)) #assume Tsym=1
        B = np.divide(np.cos(np.pi*alpha*t),1-(2*alpha*t)**2)
        p = A*B
    #Handle singularities
    p[np.argwhere(np.isnan(p))] = 1 # singularity at p(t=0)
    # singularity at t = +/- Tsym/2alpha
    p[np.argwhere(np.isinf(p))] = (alpha/2)*np.sin(np.divide(np.pi,(2*alpha)))
    return p

def raisedCosineDemo():
    """
    Raised Cosine pulses and their manifestation in frequency domain
    Usage:
        >> from pulseshapers import raisedCosineDemo
        >> raisedCosineDemo()
    """
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft, fftshift
    
    Tsym = 1 # Symbol duration in seconds
    L = 32 # oversampling rate, each symbol contains L samples
    span = 10 # filter span in symbols
    alphas= [0, 0.3, 0.5, 1] # RC roll-off factors - valid range 0 to 1
    Fs = L/Tsym # sampling frequency
    
    lineColors = ['b','r','g','k']
    fig, (ax1,ax2) = plt.subplots(1, 2)
    
    for i, alpha in enumerate(alphas):    
        b = raisedCosineDesign(alpha,span,L) # RC Pulse design
        
        # time base for symbol duration
        t = Tsym* np.arange(-span/2, span/2 + 1/L, 1/L)
        
        # plot time domain view
        ax1.plot(t,b,lineColors[i],label=r'$\alpha$='+str(alpha))
        
        #Compute FFT and plot double sided frequency domain view
        NFFT = 1<<(len(b)-1).bit_length() #Set FFT length = nextpower2(len(b))
        vals = fftshift(fft(b,NFFT))
        freqs = Fs* np.arange(-NFFT/2,NFFT/2)/NFFT
        ax2.plot(freqs,abs(vals)/abs(vals[len(vals)//2]),\
                 lineColors[i],label=r'$\alpha$='+str(alpha))
    
    ax1.set_title('Raised cosine pulse');
    ax2.set_title('Frequency response')
    ax1.legend();ax2.legend()
    fig.show()