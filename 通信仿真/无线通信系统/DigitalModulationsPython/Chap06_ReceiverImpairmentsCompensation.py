#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:40:01 2025

@author: jack
"""

#%% Program 83: DigiCommPy\chapter 6\rf impairments.py: Visualizing receiver impairments
import sys
sys.path.append("..")
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from numpy import real, imag
from DigiCommPy.modem import QAMModem #QAM Modem model
from DigiCommPy.impairments import ImpairmentModel #Impairment Model


M = 64 # M-QAM modulation order
nSym = 1000 # To generate random symbols

# uniform random symbols from 0 to M-1
inputSyms = np.random.randint(low=0, high = M, size=nSym)
modem = QAMModem(M) #initialize the M-QAM modem object
s = modem.modulate(inputSyms) #modulated sequence

impModel_1 =  ImpairmentModel(g=0.8) # gain mismatch only model
impModel_2 =  ImpairmentModel(phi=12) # phase mismatch only model
impModel_3 =  ImpairmentModel(dc_i=0.5,dc_q=0.5) # DC offsets only
impModel_4 =  ImpairmentModel(g=0.8,phi=12,dc_i=0.5,dc_q=0.5) # All impairments

#Add impairments to the input signal sequence using the models
r1 = impModel_1.receiver_impairments(s)
r2 = impModel_2.receiver_impairments(s)
r3 = impModel_3.receiver_impairments(s)
r4 = impModel_4.receiver_impairments(s)

fig, ax = plt.subplots(nrows=2, ncols = 2, figsize=(10, 8), constrained_layout=True)

ax[0,0].plot(real(s),imag(s),'b.')
ax[0,0].plot(real(r1),imag(r1),'r.');ax[0,0].set_title('IQ Gain mismatch only')

ax[0,1].plot(real(s),imag(s),'b.')
ax[0,1].plot(real(r3),imag(r3),'r.');ax[0,1].set_title('DC offsets only')

ax[1,0].plot(real(s),imag(s),'b.')
ax[1,0].plot(real(r2),imag(r2),'r.');ax[1,0].set_title('IQ Phase mismatch only')

ax[1,1].plot(real(s),imag(s),'b.')
ax[1,1].plot(real(r4),imag(r4),'r.')
ax[1,1].set_title('IQ impairments & DC offsets')
fig.show()
plt.show()
plt.close()


#%% Program 84: DigiCommPy\chapter 6\mqam awgn iq imb.py: Performance of M-QAM modulation technique with receiver impairments

import numpy as np # for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.modem import QAMModem #QAM Modem model
from DigiCommPy.channels import awgn
from DigiCommPy.impairments import ImpairmentModel #Impairment Model
from DigiCommPy.compensation import dc_compensation,blind_iq_compensation,PilotEstComp
from DigiCommPy.errorRates import ser_awgn

# ---------Input Fields------------------------
nSym=100000 # Number of input symbols
EbN0dBs = np.arange(start=-4, stop=24, step=2) # Define EbN0dB range for simulation
M = 16 # M-QAM modulation order
g=0.9
phi=8
dc_i=1.9
dc_q=1.7 # receiver impairments
# ----------------------------------------------
k = np.log2(M)
EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation

SER_1 = np.zeros(len(EbN0dBs)) # Symbol Error rates (No compensation)
SER_2 = np.zeros(len(EbN0dBs)) # Symbol Error rates (DC compensation only)
SER_3 = np.zeros(len(EbN0dBs)) # Symbol Error rates (DC comp & Blind IQ comp)
SER_4 = np.zeros(len(EbN0dBs)) # Symbol Error rates (DC comp & Pilot IQ comp)

d = np.random.randint(low=0, high = M, size=nSym) # random symbols from 0 to M-1
modem = QAMModem(M) #initialize the M-QAM modem object
modulatedSyms = modem.modulate(d) #modulated sequence

for i,EsN0dB in enumerate(EsN0dBs):
    receivedSyms = awgn(modulatedSyms,EsN0dB) # add awgn nois
    impObj = ImpairmentModel(g, phi, dc_i, dc_q) # init impairments model
    y1 = impObj.receiver_impairments(receivedSyms) # add impairments

    y2 = dc_compensation(y1) # DC compensation

    #Through Blind IQ compensation after DC compensation
    y3 = blind_iq_compensation(y2)

    #Through Pilot estimation and compensation model
    pltEstCompObj = PilotEstComp(impObj) #initialize
    y4 = pltEstCompObj.pilot_iqImb_compensation(y1) #call function


    # Enable this section - if you want to plot constellation diagram
    fig1, ax = plt.subplots(nrows = 1,ncols = 1)
    ax.plot(np.real(y1), np.imag(y1),'r.')
    ax.plot(np.real(y4), np.imag(y4),'b*')
    ax.set_title('$E_b/N_0$={} (dB)'.format(EbN0dBs[i]))
    plt.show()
    plt.close()

    # -------IQ Detectors--------
    dcap_1 = modem.iqDetector(y1) # No compensation
    dcap_2 = modem.iqDetector(y2) # DC compensation only
    dcap_3 = modem.iqDetector(y3) # DC & blind IQ comp.
    dcap_4 = modem.iqDetector(y4) # DC & pilot IQ comp.

    # ------ Symbol Error Rate Computation-------
    SER_1[i] = np.sum((d!=dcap_1))/nSym
    SER_2[i] = np.sum((d!=dcap_2))/nSym
    SER_3[i] = np.sum((d!=dcap_3))/nSym
    SER_4[i] = np.sum((d!=dcap_4))/nSym

SER_theory = ser_awgn(EbN0dBs, 'QAM', M) # theory SER
fig2, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8), constrained_layout = True)
ax.semilogy(EbN0dBs,SER_1,'*-r',label='No compensation')
ax.semilogy(EbN0dBs,SER_2,'o-b',label='DC comp only')
ax.semilogy(EbN0dBs,SER_3,'x-g',label='Sim- DC and blind iq comp')
ax.semilogy(EbN0dBs,SER_4,'D-m',label='Sim- DC and pilot iq comp')
ax.semilogy(EbN0dBs, SER_theory,'k',ls = '--', label='Theoretical')
ax.set_ylim([1e-4, 1.1])
ax.set_xlabel('$E_b/N_0$ (dB)');ax.set_ylabel('Symbol Error Rate ($P_s$)')
ax.set_title(f'Probability of Symbol Error {M}-QAM signals');
ax.legend()
fig2.show()




























