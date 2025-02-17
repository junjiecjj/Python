function [meanDelay,rmsDelay,symbolRate,coherenceBW] = meas_continuous_PDP(fun,lowerLim,upperLim)
%Function to calculate mean Delay, RMS delay spread, maximum symbol
%rate that a signal can be transmitted without ISI and the coherence
%BW for the PDP equation specified as function handle(fun) 
%  example: fun = @(tau) exp(-tau/0.00001); %given PDP equation
%lowerLim - lower limit for integration
%upperLim - upper limit  for integration
moment_1 = @(x) x.*fun(x);
meanDelay = integral(moment_1,lowerLim,upperLim)/integral(fun,lowerLim,upperLim);
moment_2 = @(y) ((y-meanDelay).^2).*fun(y);
rmsDelay = sqrt(integral(moment_2,lowerLim,upperLim)/integral(fun,lowerLim,upperLim));
symbolRate = 1/(10*rmsDelay); %maximum symbol rate to avoid ISI
coherenceBW = 1/(50*rmsDelay);%for 0.9 correlation
%coherenceBW = 1/(5*rmsDelay);%for 0.5 correlation
end
