"""
Script: DigiCommPy.chapter_6.mqam_awgn_iqimb.py
Eb/N0 Vs SER for M-QAM modulation with receiver impairments

@author: Mathuranathan Viswanathan
Created on Sep 6, 2019
"""
import sys
sys.path.append("..")
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
M=64 # M-QAM modulation order
g=0.9; phi=8; dc_i=1.9; dc_q=1.7 # receiver impairments
# ----------------------------------------------
k=np.log2(M)
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
    impObj =  ImpairmentModel(g,phi,dc_i,dc_q) # init impairments model
    y1 = impObj.receiver_impairments(receivedSyms) # add impairments

    y2 = dc_compensation(y1) # DC compensation

    #Through Blind IQ compensation after DC compensation
    y3 = blind_iq_compensation(y2)

    #Through Pilot estimation and compensation model
    pltEstCompObj = PilotEstComp(impObj) #initialize
    y4 = pltEstCompObj.pilot_iqImb_compensation(y1) #call function

    """
    Enable this section - if you want to plot constellation diagram
    fig1, ax = plt.subplots(nrows = 1,ncols = 1)
    ax.plot(np.real(y1),np.imag(y1),'r.')
    ax.plot(np.real(y4),np.imag(y4),'b*')
    ax.set_title('$E_b/N_0$={} (dB)'.format(EbN0dBs[i]));
    fig1.show();input()
    """
    # -------IQ Detectors--------
    dcap_1 = modem.iqDetector(y1) # No compensation
    dcap_2 = modem.iqDetector(y2) # DC compensation only
    dcap_3 = modem.iqDetector(y3) # DC & blind IQ comp.
    dcap_4 = modem.iqDetector(y4) # DC & pilot IQ comp.

    # ------ Symbol Error Rate Computation-------
    SER_1[i]=sum((d!=dcap_1))/nSym; SER_2[i]=sum((d!=dcap_2))/nSym
    SER_3[i]=sum((d!=dcap_3))/nSym; SER_4[i]=sum((d!=dcap_4))/nSym

SER_theory = ser_awgn(EbN0dBs,'QAM',M) #theory SER
fig2, ax = plt.subplots(nrows = 1,ncols = 1)
ax.semilogy(EbN0dBs,SER_1,'*-r',label='No compensation')
ax.semilogy(EbN0dBs,SER_2,'o-b',label='DC comp only')
ax.semilogy(EbN0dBs,SER_3,'x-g',label='Sim- DC and blind iq comp')
ax.semilogy(EbN0dBs,SER_4,'D-m',label='Sim- DC and pilot iq comp')
#ax.semilogy(EbN0dBs,SER_theory,'k',label='Theoretical')
ax.set_xlabel('$E_b/N_0$ (dB)');ax.set_ylabel('Symbol Error Rate ($P_s$)')
ax.set_title('Probability of Symbol Error 64-QAM signals');
ax.legend();fig2.show()
