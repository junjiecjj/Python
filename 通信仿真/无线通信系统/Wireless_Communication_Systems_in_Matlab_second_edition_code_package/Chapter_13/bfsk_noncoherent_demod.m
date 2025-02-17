function a_cap = bfsk_noncoherent_demod(r,Fc,Fd,L,Fs)
%Non-coherent demodulation of BFSK modulated signal
%r - BFSK modulated signal at the receiver 
%Fc - center frequency of the carrier in Hertz
%Fd - frequency separation measured from Fc
%L - number of samples in 1-bit period
%Fs - Sampling frequency for discrete-time simulation
%a_cap - data bits after demodulation
t = (0:1:length(r)-1)/Fs; %time base
F1 = (Fc+Fd/2); F2 = (Fc-Fd/2);

%define four basis functions
p1c = cos(2*pi*F1*t);  p2c = cos(2*pi*F2*t);
p1s = -sin(2*pi*F1*t); p2s = -sin(2*pi*F2*t);

%multiply and integrate from 0 to Tb
r1c = conv(r.*p1c,ones(1,L)); r2c = conv(r.*p2c,ones(1,L)); 
r1s = conv(r.*p1s,ones(1,L)); r2s = conv(r.*p2s,ones(1,L)); 

%sample at every sampling instant
r1c = r1c(L:L:end); r2c = r2c(L:L:end);
r1s = r1s(L:L:end); r2s = r2s(L:L:end);

x = r1c.^2 + r1s.^2; y = r2c.^2 + r2s.^2;%square and add

a_cap=(x-y)>0; %compare and decide