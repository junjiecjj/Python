>> Ps=[-20 -10 -10 0];%list of power values in dBm
>> TAUs =[0 1e-6 2e-6 5e-6];%list of taus in the discrete PDP
>> [meanDelay,rmsDelay,symbolRate,coherenceBW] = meas_discrete_PDP(Ps,TAUs)
>> meanDelay  = 4.38e-06
   rmsDelay   = 1.37e-06
   symbolRate = 7.27e+04
   coherenceBW = 1.4554e+04