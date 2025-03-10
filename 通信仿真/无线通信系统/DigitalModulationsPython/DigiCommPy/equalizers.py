"""
Module: DigiCommPy.equalizers.py

@author: Mathuranathan Viswanathan
Created on Aug 22, 2019
"""
import numpy as np
import abc

class Equalizer():
    # Base class: Equalizer (Abstract base class)
    # Attribute definitions:
    #    self.N: length of the equalizer
    #    self.w : equalizer weights
    #    self.delay : optimized equalizer delay
    def __init__(self,N): # constructor for N tap FIR equalizer
        self.N = N
        self.w = np.zeros(N)
        self.opt_delay = 0
        
    @abc.abstractmethod
    def design(self): #Abstract method
        "Design the equalizer for the given impulse response and SNR"
    
    def convMatrix(self,h,p):
        """
        Construct the convolution matrix of size (N+p-1)x p from the
        input matrix h of size N. (see chapter 1)
        Parameters:
            h : numpy vector of length L
            p : scalar value
        Returns:
            H : convolution matrix of size (L+p-1)xp
        """
        col=np.hstack((h,np.zeros(p-1)))
        row=np.hstack((h[0],np.zeros(p-1)))
        
        from scipy.linalg import toeplitz
        H=toeplitz(col,row)
        return H
        
    def equalize(self,inputSamples):
        """
        Equalize the given input samples and produces the output
        Parameters:
            inputSamples : signal to be equalized
        Returns:
            equalizedSamples: equalized output samples
        """
        #convolve input with equalizer tap weights
        equalizedSamples = np.convolve(inputSamples,self.w)
        return equalizedSamples    
        
class zeroForcing(Equalizer): #Class zero-forcing equalizer
    def design(self,h,delay=None): #override method in Equalizer abstract class
        """
        Design a zero forcing equalizer for given channel impulse response (CIR).
        If the tap delay is not given, a delay optimized equalizer is designed
        Parameters:
            h : channel impulse response
            delay: desired equalizer delay (optional)
        Returns: MSE: Mean Squared Error for the designed equalizer
        """
        L = len(h)
        H = self.convMatrix(h,self.N) #(L+N-1)xN matrix - see Chapter 1
        # compute optimum delay based on MSE
        Hp = np.linalg.pinv(H) #Moore-Penrose Pseudo inverse
        #get index of maximum value using argmax, @ for matrix multiply
        opt_delay = np.argmax(np.diag(H @ Hp))
        self.opt_delay = opt_delay #optimized delay
        
        if delay==None:
            delay=opt_delay
        elif delay >=(L+self.N-1):
            raise ValueError('Given delay is too large delay (should be < L+N-1')
        
        k0 = delay
        d=np.zeros(self.N+L-1);d[k0]=1 #optimized position of equalizer delay
        self.w=Hp @ d # Least Squares solution, @ for matrix multiply
        MSE=(1-d.T @ H @ Hp @ d) #MSE and err are equivalent,@ for matrix multiply
        return MSE

class MMSEEQ(Equalizer): #Class MMSE Equalizer
    def design(self,h,snr,delay=None): #override method in Equalizer abstract class
        """
        Design a MMSE equalizer for given channel impulse response (CIR) and
        signal to noise ratio (SNR). If the tap delay is not given, a delay
        optimized equalizer is designed
        Parameters:
            h : channel impulse response
            snr: input signal to noise ratio in dB scale
            delay: desired equalizer delay (optional)
        Returns: MSE: Mean Squared Error for the designed equalizer
        """
        L = len(h)
        H=self.convMatrix(h,self.N) #(L+N-1)xN matrix - see Chapter 1
        gamma = 10**(-snr/10) # inverse of SNR
        # compute optimum delay
        opt_delay = np.argmax(np.diag(H @ np.linalg.inv(H.T @ H+gamma * np.eye(self.N))@ H.T)) # @ for matrix multiply
        self.opt_delay = opt_delay #optimized delay
        
        if delay==None:
            delay=opt_delay
        if delay >=(L+self.N-1):
            raise ValueError('Given delay is too large delay (should be < L+N-1')
        
        k0 = delay
        d=np.zeros(self.N+L-1)
        d[k0]=1 # optimized position of equalizer delay
        # Least Squares solution, @ for matrix multiply
        self.w=np.linalg.inv(H.T @ H+ gamma * np.eye(self.N))@ H.T @ d
        # assume var(a)=1, @ for matrix multiply
        MSE=(1-d.T @ H @ np.linalg.inv(H.T @ H+gamma * np.eye(self.N)) @ H.T @ d)
        return MSE
    
class LMSEQ(Equalizer): #Class LMS adaptive equalizer
    def design(self,mu,r,a):
        """
        Design an adaptive FIR filter using LMS update equations (Training Mode)
        Parameters:
            N : desired length of the filter
            mu : step size for the LMS update
            r : received/input sequence
            a: reference sequence
        """
        N =self.N
        w = np.zeros(N)
        for k in range(N, len(r)):
            r_vector = r[k:k-N:-1]
            e = a[k] - w @ r_vector.T # @ denotes matrix multiplication
            w = w + mu * e * r_vector # @ denotes matrix multiplication
        self.w = w #set the final filter coefficients    