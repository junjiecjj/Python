function [p,t,filtDelay]=rectFunction(L,Nsym)
%Function for generating rectangular pulse for the given inputs
%L - oversampling factor (number of samples per symbol)
%Nsym - filter span in symbol durations
%Returns the output pulse p(t) that spans the discrete-time base
%-Nsym:1/L:Nsym. Also returns the filter delay.
Tsym=1;t=-(Nsym/2):1/L:(Nsym/2);%unit symbol duration time-base
p=(t > -Tsym/2) .* (t <= Tsym/2);%rectangular pulse
%FIR filter delay = (N-1)/2, N=length of the filter
filtDelay =  (length(p)-1)/2; %FIR filter delay
end