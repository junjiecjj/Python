function [p,t,filtDelay]=sincFunction(L,Nsym)
%Function for generating sinc function for the given inputs
%L - oversampling factor (number of samples per symbol)
%Nsym - filter span in symbol durations
%Returns the output pulse p(t) that spans the discrete-time base
%-Nsym:1/L:Nsym. Also returns the filter delay when the function 
%is viewed as an FIR filter
Tsym=1;t=-(Nsym/2):1/L:(Nsym/2);%unit symbol duration time-base
p = sin(pi*t/Tsym)./(pi*t/Tsym);
p(ceil(length(p)/2))=1; %catch sinc(0/0) condition
%FIR filter delay = (N-1)/2, N=length of the filter
filtDelay =  (length(p)-1)/2; %FIR filter delay
end