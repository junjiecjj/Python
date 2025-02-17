function [p,t,filtDelay]=srrcFunction(beta,L,Nsym)
%Function for generating square-root raised-cosine (SRRC) pulse
% beta - roll-off factor of SRRC pulse,
% L - oversampling factor (number of samples per symbol)
% Nsym - filter span in symbol durations
%Returns the output pulse p(t) that spans the discrete-time base
%-Nsym:1/L:Nsym. Also returns the filter delay when the function 
%is viewed as an FIR filter
Tsym=1; t=-(Nsym/2):1/L:(Nsym/2);%unit symbol duration time-base

num = sin(pi*t*(1-beta)/Tsym)+...
    ((4*beta*t/Tsym).*cos(pi*t*(1+beta)/Tsym));
den = pi*t.*(1-(4*beta*t/Tsym).^2)/Tsym;
p = 1/sqrt(Tsym)*num./den; %srrc pulse definition

%handle catch corner cases (singularities)
p(ceil(length(p)/2))=1/sqrt(Tsym)*((1-beta)+4*beta/pi);
temp=(beta/sqrt(2*Tsym))*( (1+2/pi)*sin(pi/(4*beta)) ...
    + (1-2/pi)*cos(pi/(4*beta)));
p(t==Tsym/(4*beta))=temp; p(t==-Tsym/(4*beta))=temp;
%FIR filter delay = (N-1)/2, N=length of the filter
filtDelay  =  (length(p)-1)/2; %FIR filter delay
end