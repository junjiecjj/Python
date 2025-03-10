"""
Essential functions described in Chapter 1

@author: Mathuranathan Viswanathan
Created on Jul 15, 2019
"""
import numpy as np

def plotWelchPSD(x,fs,fc,ax = None,color='b', label=None):
    """
    Plot PSD of a carrier modulated signal using Welch estimate
    Parameters:
        x : signal vector (numpy array) for which the PSD is plotted
        fs : sampling Frequency
        fc : center carrier frequency of the signal
        ax : Matplotlib axes object reference for plotting
        color : color character (format string) for the plot
    """
    from scipy.signal import hanning, welch
    from numpy import log10
    nx = max(x.shape)
    na = 16 # averaging factor to plot averaged welch spectrum
    w = hanning(nx//na) #// is for integer floor division
    # Welch PSD estimate with Hanning window and no overlap
    f, Pxx =welch(x,fs,window = w,noverlap=0)
    indices = (f>=fc) & (f<4*fc)   # To plot PSD from Fc to 4*Fc
    Pxx = Pxx[indices]/Pxx[indices][0] # normalized psd w.r.t Fc
    ax.plot(f[indices]-fc,10*log10(Pxx),color,label=label) #Plot in the given axes
    
def conv_brute_force(x,h):
    """
    Brute force method to compute convolution
    Parameters:
        x, h : numpy vectors
    Returns:
        y : convolution of x and h
    """
    N=len(x)
    M=len(h)
    y = np.zeros(N+M-1) #array filled with zeros
    for i in np.arange(0,N):
        for j in np.arange(0,M):
            y[i+j] = y[i+j] + x[i] * h[j]
    return y

def convMatrix(h,p):
    """
    Construct the convolution matrix of size (N+p-1)x p from the input matrix h of size N.
    Parameters:
        h : numpy vector of length N
        p : scalar value
    Returns:
        H : convolution matrix of size (N+p-1)xp
    """
    col=np.hstack((h,np.zeros(p-1)))
    row=np.hstack((h[0],np.zeros(p-1)))
    
    from scipy.linalg import toeplitz
    H=toeplitz(col,row)
    return H

def my_convolve(h,x):
    """
    Convolve two sequences h and x of arbitrary lengths: y=h*x
    Parameters:
        h,x : numpy vectors
    Returns:
        y : convolution of h and x
    """
    H=convMatrix(h,len(x)) #see convMatrix function
    y=H @ x.transpose() # equivalent to np.convolve(h,x) function
    return y

def analytic_signal(x):
    """
    Generate analytic signal using frequency domain approach
    Parameters:
        x : Signal data. Must be real
    Returns:
        z : Analytic signal of x
    """
    from scipy.fftpack import fft,ifft
    N = len(x)
    X = fft(x,N)    
    Z = np.hstack((X[0], 2*X[1:N//2], X[N//2], np.zeros(N//2-1)))
    z = ifft(Z,N)
    return z