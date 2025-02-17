function [PL,Pr_dBm] = logNormalShadowing(Pt_dBm,Gt_dBi,Gr_dBi,f,d0,d,L,sigma,n)
%Pt_dBm = Transmitted power in dBm
%Gt_dBi = Gain of the Transmitted antenna in dBi
%Gr_dBi = Gain of the Receiver antenna in dBi
%f = frequency of transmitted signal in Hertz
%d0 = reference distance of receiver from the transmitter in meters
%d = array of distances at which the loss needs to be calculated
%L = Other System Losses, for no Loss case L=1
%sigma = Standard deviation of log Normal distribution (in dB)
%n = path loss exponent
%Pr_dBm = Received power in dBm
%PL = path loss due to log normal shadowing
lambda=3*10^8/f; %Wavelength in meters
K = 20*log10(lambda/(4*pi)) - 10*n*log10(d0) - 10*log10(L);%path-loss factor
X = sigma*randn(1,numel(d)); %normal random variable
PL = Gt_dBi + Gr_dBi + K -10*n*log10(d/d0) - X ;%PL(d) including antennas gains
Pr_dBm = Pt_dBm + PL; %Receieved power in dBm at d meters
end

