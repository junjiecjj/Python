"""
Script: DigiCommPy.chapter_6.rf_impairments.py
Visualize receiver impairments in a complex plane

@author: Mathuranathan Viswanathan
Created on Sep 6, 2019
"""
import sys
sys.path.append("..")
import numpy as np #for numerical computing
from numpy import real,imag
from DigiCommPy.modem import QAMModem #QAM Modem model
from DigiCommPy.impairments import ImpairmentModel #Impairment Model
import matplotlib.pyplot as plt #for plotting functions

M=64 # M-QAM modulation order
nSym=1000 # To generate random symbols

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

fig, ax = plt.subplots(nrows=2,ncols = 2)

ax[0,0].plot(real(s),imag(s),'b.')
ax[0,0].plot(real(r1),imag(r1),'r.');ax[0,0].set_title('IQ Gain mismatch only')

ax[0,1].plot(real(s),imag(s),'b.')
ax[0,1].plot(real(r3),imag(r3),'r.');ax[0,1].set_title('DC offsets only')

ax[1,0].plot(real(s),imag(s),'b.')
ax[1,0].plot(real(r2),imag(r2),'r.');ax[1,0].set_title('IQ Phase mismatch only')

ax[1,1].plot(real(s),imag(s),'b.')
ax[1,1].plot(real(r4),imag(r4),'r.')
ax[1,1].set_title('IQ impairments & DC offsets');fig.show()
