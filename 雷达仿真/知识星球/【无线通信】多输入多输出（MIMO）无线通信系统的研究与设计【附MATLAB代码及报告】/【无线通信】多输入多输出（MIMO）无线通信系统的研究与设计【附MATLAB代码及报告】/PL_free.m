function PL=PL_free(fc,d,Gt,Gr)
% Free Space Path Loss Model 
% Inputs: fc : Carrier frequency[Hz]
% d : Distance between base station and mobile station[m] 
% Gt/Gr : Transmitter/Receiver gain
% Output: PL : Path loss[dB] 
lamda = 3e8/fc;
tmp = lamda./(4*pi*d);
if nargin>2, tmp = tmp*sqrt(Gt); end 
if nargin>3, tmp = tmp*sqrt(Gr); end 
PL = -20*log10(tmp); % Eq.(1.2)/(1.3)
