function [Pr_dBm,PL_dB] = FriisModel(Pt_dBm,Gt_dBi,Gr_dBi,f,d,L,n)
%Pt_dBm = Transmitted power in dBm
%Gt_dBi = Gain of the Transmitted antenna in dBi
%Gr_dBi = Gain of the Receiver antenna in dBi
%f = frequency of transmitted signal in Hertz
%d = array of distances at which the loss needs to be calculated
%L = Other System Losses, No Loss case L=1
%n = path loss exponent (n=2 for free space)
%Pr_dBm = Received power in dBm
%PL_dB  = constant path loss factor (including antenna gains)
lambda=3*10^8/f; %Wavelength in meters
PL_dB = Gt_dBi + Gr_dBi + 20*log10(lambda/(4*pi))-10*n*log10(d)-10*log10(L);%
Pr_dBm = Pt_dBm + PL_dB; %Receieved power in dBm at d meters
end