function [p,t,filtDelay]=raisedCosineFunction(alpha,L,Nsym)
%Function for generating raised-cosine (RC) pulse
% alpha - roll-off factor,L - oversampling factor, 
% Nsym - filter span in symbols
%Returns the output pulse p(t) that spans the discrete-time base
%-Nsym:1/L:Nsym.Also returns the filter delay when the function 
%is viewed as an FIR filter
Tsym=1; t=-(Nsym/2):1/L:(Nsym/2);%unit symbol duration time-base
A = sin(pi*t/Tsym)./(pi*t/Tsym); B=cos(pi*alpha*t/Tsym);
p = A.*B./(1-(2*alpha*t/Tsym).^2);
p(ceil(length(p)/2))=1;%p(0)=1 & p(0) occurs exactly at the center
temp=(alpha/2)*sin(pi/(2*alpha)); %p(t=+-1/(2a))=(a/2)sin(pi/(2a))
p(t==Tsym/(2*alpha))=temp; p(t==-Tsym/(2*alpha))=temp;
%FIR filter delay = (N-1)/2, N=length of the filter
filtDelay =  (length(p)-1)/2; %FIR filter delay
end