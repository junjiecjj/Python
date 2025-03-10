"""
Script: DigiCommPy.chapter_4.ricianPerformance.py
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
from DigiCommPy.channels import awgn,ricianFading
from DigiCommPy.errorRates import ser_rician

#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start=0,stop = 22, step = 2) # Eb/N0 range in dB for simulation
K_dBs = [3,5,10,20] # array of K factors for Rician fading in dB
mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM'
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

    SER_theory = ser_rician(K_dB,EbN0dBs,mod_type,M)
    ax.semilogy(EbN0dBs,SER_sim,color = colors[i],marker='o',linestyle='',label='Sim K='+str(K_dB)+' dB')
    ax.semilogy(EbN0dBs,SER_theory,color = colors[i],linestyle='-',label='Theory K='+str(K_dB)+' dB')

ax.set_xlabel('Eb/N0(dB)');ax.set_ylabel('SER ($P_s$)')
ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over Rayleigh flat fading channel');ax.legend();fig.show()
