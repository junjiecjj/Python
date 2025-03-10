"""
Script: DigiCommPy.chapter_4.rayleighPerformance.py
Eb/N0 Vs SER for PSK/QAM/PAM/FSK over AWGN (complex baseband model)

@author: Mathuranathan Viswanathan
Created on Aug 8, 2019
"""
import sys
sys.path.append("..")
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from matplotlib import cm # colormap for color palette
from scipy.special import erfc
from DigiCommPy.modem import PSKModem,QAMModem,PAMModem,FSKModem
from DigiCommPy.channels import awgn,rayleighFading
from DigiCommPy.errorRates import ser_rayleigh

#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start=-4,stop = 12, step = 2) # Eb/N0 range in dB for simulation
mod_type = 'PAM' # Set 'PSK' or 'QAM' or 'PAM
arrayOfM = [2,4,8,16,32] # array of M values to simulate
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
ax.legend();fig.show()
