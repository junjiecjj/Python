function q = LinearPhase_Gen_DDM(Nt, N_c, flag)
%% DDMA, Divide the entire doppler into Nt+1 or Nt parts
% input
%       Nt      # Tx
%       N_c     # slow time
%       flag    indicator to divide the total N_c into Nt parts or Nt+1
%               parts, 
%               1: Nt+1
%               2: Nt
% output
%       q       the code in each Tx antenna(Nt × N_c)
%% Nt+1 parts
if flag == 1
    % index=0 is retained to introduce difference    
    index = N_c/(Nt+1) : N_c/(Nt+1) : N_c-N_c/(Nt+1);
    q = exp(1j*2*pi/N_c*index.'*(0:N_c-1));
else
%% Nt parts
    index = 0 : N_c/(Nt) : N_c-N_c/(Nt);    
    q = exp(1j*2*pi/N_c*index.'*(0:N_c-1));
end
end