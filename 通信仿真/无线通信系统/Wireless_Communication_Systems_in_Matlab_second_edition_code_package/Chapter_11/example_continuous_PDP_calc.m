>> fun = @(tau) exp(-tau/0.00001); %given PDP equation
>> [meanDelay,rmsDelay,symbolRate,coherenceBW] = meas_continuous_PDP(fun,0,10e-6)
>> meanDelay  = 4.18e-06
   rmsDelay   = 2.81e-06
   symbolRate = 3.55e+04
   coherenceBW = 7.1010e+03
