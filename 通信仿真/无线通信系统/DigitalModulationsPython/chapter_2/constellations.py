"""
% Constellations of RC filtered waveforms of MSK and QPSK variants

@author: Mathuranathan Viswanathan
Created on Jul 26, 2019
"""
#Execute in Python3: exec(open("chapter_2/constellations.py").read())
import numpy as np
import matplotlib.pyplot as plt
from passband_modulations import qpsk_mod,oqpsk_mod,piBy4_dqpsk_mod,msk_mod
from pulseshapers import raisedCosineDesign

N=1000 # Number of symbols to transmit, keep it small and adequate
fc=10; L=8 # carrier frequency and oversampling factor
a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's

#modulate the source symbols using QPSK,QPSK,pi/4-DQPSK and MSK
qpsk_result= qpsk_mod(a,fc,L)
oqpsk_result = oqpsk_mod(a,fc,L)
piby4qpsk_result = piBy4_dqpsk_mod(a,fc,L)
msk_result = msk_mod(a,fc,L);

#Pulse shape the modulated waveforms by convolving with RC filter
alpha = 0.3; span = 10 # RC filter alpha and filter span in symbols
b =raisedCosineDesign(alpha,span, L) # RC pulse shaper
iRC_qpsk=np.convolve(qpsk_result['I(t)'],b,mode='valid')#RC - QPSK I(t)
qRC_qpsk= np.convolve(qpsk_result['Q(t)'],b,mode='valid') #RC - QPSK Q(t)
iRC_oqpsk= np.convolve(oqpsk_result['I(t)'],b,mode='valid')#RC - OQPSK I(t)
qRC_oqpsk= np.convolve(oqpsk_result['Q(t)'],b,mode='valid')#RC - OQPSK Q(t)
iRC_piby4qpsk=np.convolve(piby4qpsk_result['U(t)'],b,mode='valid')#pi/4-QPSK I
qRC_piby4qpsk=np.convolve(piby4qpsk_result['V(t)'],b,mode='valid')#pi/4-QPSK Q
i_msk = msk_result['sI(t)'] # MSK sI(t)
q_msk = msk_result['sQ(t)'] # MSK sQ(t)

fig, axs = plt.subplots(2, 2)

axs[0,0].plot(iRC_qpsk,qRC_qpsk)# RC shaped QPSK
axs[0,1].plot(iRC_oqpsk,qRC_oqpsk)# RC shaped OQPSK
axs[1,0].plot(iRC_piby4qpsk,qRC_piby4qpsk)# RC shaped pi/4-QPSK
axs[1,1].plot(i_msk[20:-20],q_msk[20:-20])# RC shaped OQPSK

axs[0,0].set_title(r'QPSK, RC $\alpha$='+str(alpha))
axs[0,0].set_xlabel('I(t)');axs[0,0].set_ylabel('Q(t)');
axs[0,1].set_title(r'OQPSK, RC $\alpha$='+str(alpha))
axs[0,1].set_xlabel('I(t)');axs[0,1].set_ylabel('Q(t)');
axs[1,0].set_title(r'$\pi$/4 - QPSK, RC $\alpha$='+str(alpha))
axs[1,0].set_xlabel('I(t)');axs[1,0].set_ylabel('Q(t)');
axs[1,1].set_title('MSK')
axs[1,1].set_xlabel('I(t)');axs[1,1].set_ylabel('Q(t)');
fig.show()