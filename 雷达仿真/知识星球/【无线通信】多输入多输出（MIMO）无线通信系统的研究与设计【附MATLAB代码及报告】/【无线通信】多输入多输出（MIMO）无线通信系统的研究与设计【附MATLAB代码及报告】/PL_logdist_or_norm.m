function PL=PL_logdist_or_norm(fc,d,d0,n,sigma) 
% Log-distance or Log-normal shadowing path loss model 
% Inputs: fc : Carrier frequency[Hz] 
% d : Distance between base station and mobile station[m] 
% d0 : Reference distance[m] 
% n : Path loss exponent 
% sigma : Variance[dB]
lamda=3e8/fc; PL= -20*log10(lamda/(4*pi*d0))+10*n*log10(d/d0); % Eq.(1.4)
if nargin>4, PL = PL + sigma*randn(size(d)); end % Eq.(1.5)