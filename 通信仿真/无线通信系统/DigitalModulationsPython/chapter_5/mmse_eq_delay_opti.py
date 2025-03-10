"""
Script: DigiCommPy.chapter_5.mmse_eq_delay_opti.py
The test methodology takes the example given in Yu Gong et al., Adaptive MMSE equalizer with optimum tap-length and
decision delay, sspd 2010 and try to get a similar plot as given in Figure 3 of the journal paper

@author: Mathuranathan Viswanathan
Created on Aug 29, 2019
"""
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt #for plotting functions

h=np.array([-0.1, -0.3, 0.4, 1, 0.4, 0.3, -0.1]) # test channel
SNR=10 # Signal-to-noise ratio at the equalizer input in dB
Ns= np.arange(start=5, stop=35, step=5) # sweep number of equalizer taps from 5 to 30
maxDelay=Ns[-1]+len(h)-2 #max delay cannot exceed this value
optimalDelay=np.zeros(len(Ns));

from DigiCommPy.equalizers import MMSEEQ

fig, ax = plt.subplots(nrows=1,ncols = 1)
for i,N in enumerate(Ns): #sweep number of equalizer taps
    maxDelay = N+len(h)-2
    mse=np.zeros(maxDelay)
    for j,delay in enumerate(range(0,maxDelay)): # sweep delays
        # compute MSE and optimal delay for each combination
        mmse_eq = MMSEEQ(N) #initialize MMSE equalizer (object) of length N
        mse[j]=mmse_eq.design(h,SNR,delay)
        optimalDelay[i] = mmse_eq.opt_delay
    #plot mse in log scale
    ax.plot(np.arange(0,maxDelay),np.log10(mse),label='N='+str(N))
ax.set_title('MSE Vs eq. delay for given channel and equalizer lengths')
ax.set_xlabel('Equalizer delay');ax.set_ylabel('$log_{10}$[mse]');
ax.legend();fig.show()
#display optimal delays for each selected filter length N. this will correspond
#with the bottom of the buckets displayed in the plot
print('Optimal Delays for each N value ->{}'.format(optimalDelay))
