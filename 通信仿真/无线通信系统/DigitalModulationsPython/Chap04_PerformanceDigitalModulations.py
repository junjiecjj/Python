#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:37:52 2025

@author: jack
"""

import sys
sys.path.append("..")

import numpy as np # for numerical computing
import matplotlib.pyplot as plt # for plotting functions
from matplotlib import cm # colormap for color palette
from scipy.special import erfc
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn


#%%  Performance of modulations in AWGN
#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start = -4, stop = 12, step = 2) # Eb/N0 range in dB for simulation
mod_type = 'FSK' # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
#arrayOfM=[4,16,64,256] # uncomment this line if MOD_TYPE='QAM'
coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem,'fsk':FSKModem}
colors = plt.cm.jet(np.linspace(0, 1, len(arrayOfM))) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1)

for i, M in enumerate(arrayOfM):
    #-----Initialization of various parameters----
    k=np.log2(M)
    EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation
    SER_sim = np.zeros(len(EbN0dBs)) # simulated Symbol error rates
    inputSyms = np.random.randint(low=0, high = M, size=nSym)
    # uniform random symbols from 0 to M-1

    if mod_type.lower()=='fsk':
        modem=modem_dict[mod_type.lower()](M,coherence)#choose modem from dictionary
    else: #for all other modulations
        modem = modem_dict[mod_type.lower()](M)#choose modem from dictionary
    modulatedSyms = modem.modulate(inputSyms) #modulate

    for j,EsN0dB in enumerate(EsN0dBs):
        receivedSyms = awgn(modulatedSyms,EsN0dB) #add awgn noise

        if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
            detectedSyms = modem.demodulate(receivedSyms,coherence)
        else: #demodulate (Refer Chapter 3)
            detectedSyms = modem.demodulate(receivedSyms)

        SER_sim[j] = np.sum(detectedSyms != inputSyms)/nSym

    SER_theory = ser_awgn(EbN0dBs, mod_type, M, coherence) #theory SER
    ax.semilogy(EbN0dBs, SER_sim, color = colors[i], marker='o', linestyle='', label='Sim '+str(M)+'-'+mod_type.upper())
    ax.semilogy(EbN0dBs, SER_theory, color = colors[i], linestyle='-', label='Theory '+str(M)+'-'+mod_type.upper())

ax.set_xlabel('Eb/N0(dB)')
ax.set_ylabel('SER ($P_s$)')
ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over AWGN')
ax.legend()
fig.show()


#%%  Performance in Rayleigh ﬂat fading

import sys
sys.path.append("..")
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from matplotlib import cm # colormap for color palette
from scipy.special import erfc
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn, rayleighFading
from DigiCommPy.errorRates import ser_rayleigh

#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start=-4,stop = 12, step = 2) # Eb/N0 range in dB for simulation
mod_type = 'PAM' # Set 'PSK' or 'QAM' or 'PAM
arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
#arrayOfM=[4,16,64,256] # uncomment this line if MOD_TYPE='QAM'

modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem}
colors = plt.cm.jet(np.linspace(0,1,len(arrayOfM))) # colormap
fig, ax = plt.subplots(nrows=1,ncols = 1)

for i, M in enumerate(arrayOfM):
    k=np.log2(M)
    EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation
    SER_sim = np.zeros(len(EbN0dBs)) # simulated Symbol error rates
    # uniform random symbols from 0 to M-1
    inputSyms = np.random.randint(low=0, high = M, size=nSym)

    modem = modem_dict[mod_type.lower()](M)#choose a modem from the dictionary
    modulatedSyms = modem.modulate(inputSyms) #modulate

    for j,EsN0dB in enumerate(EsN0dBs):
        h_abs = rayleighFading(nSym) #Rayleigh flat fading samples
        hs = h_abs*modulatedSyms #fading effect on modulated symbols
        receivedSyms = awgn(hs,EsN0dB) #add awgn noise

        y = receivedSyms/h_abs # decision vector
        detectedSyms = modem.demodulate(y) #demodulate (Refer Chapter 3)
        SER_sim[j] = np.sum(detectedSyms != inputSyms)/nSym

    SER_theory = ser_rayleigh(EbN0dBs,mod_type,M) #theory SER
    ax.semilogy(EbN0dBs,SER_sim,color = colors[i],marker='o',linestyle='',label='Sim '+str(M)+'-'+mod_type.upper())
    ax.semilogy(EbN0dBs,SER_theory,color = colors[i],linestyle='-',label='Theory '+str(M)+'-'+mod_type.upper())

ax.set_xlabel('Eb/N0(dB)');ax.set_ylabel('SER ($P_s$)')
ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over Rayleigh flat fading channel')
ax.legend()
fig.show()



#%%  Performance over Rician ﬂat fading

import sys
sys.path.append("..")
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from matplotlib import cm # colormap for color palette
from scipy.special import erfc
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn, ricianFading
from DigiCommPy.errorRates import ser_rician

#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start = 0, stop = 22, step = 2) # Eb/N0 range in dB for simulation
K_dBs = [3, 5, 10, 20] # array of K factors for Rician fading in dB
mod_type = 'QAM' # Set 'PSK' or 'QAM' or 'PAM'
M = 4 # M value for the modulation to simulate

modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem}
colors = plt.cm.jet(np.linspace(0,1,len(K_dBs))) # colormap
fig, ax = plt.subplots(nrows=1,ncols = 1)

for i, K_dB in enumerate(K_dBs):
    #-----Initialization of various parameters----
    k=np.log2(M)
    EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation
    SER_sim = np.zeros(len(EbN0dBs)) # simulated Symbol error rates
    # uniform random symbols from 0 to M-1
    inputSyms = np.random.randint(low=0, high = M, size=nSym)

    modem = modem_dict[mod_type.lower()](M)#choose a modem from the dictionary
    modulatedSyms = modem.modulate(inputSyms) #modulate

    for j,EsN0dB in enumerate(EsN0dBs):
        h_abs = ricianFading(K_dB,nSym) #Rician flat fading samples
        hs = h_abs*modulatedSyms #fading effect on modulated symbols
        receivedSyms = awgn(hs,EsN0dB) #add awgn noise
        y = receivedSyms/h_abs # decision vector
        detectedSyms = modem.demodulate(y) #demodulate (Refer Chapter 3)
        SER_sim[j] = np.sum(detectedSyms != inputSyms)/nSym

    SER_theory = ser_rician(K_dB, EbN0dBs, mod_type, M)
    ax.semilogy(EbN0dBs,SER_sim,color = colors[i],marker='o',linestyle='',label='Sim K='+str(K_dB)+' dB')
    ax.semilogy(EbN0dBs,SER_theory,color = colors[i],linestyle='-',label='Theory K='+str(K_dB)+' dB')

ax.set_xlabel('Eb/N0(dB)')
ax.set_ylabel('SER ($P_s$)')
ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over Rayleigh flat fading channel')
ax.legend()
fig.show()





















