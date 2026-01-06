function X2_Detected = RDMsetZero(X2_Detected, col, N_c, Nt)
% Set the processed position to 0
% 
% input
%       X2_Detected                 0 or 1 after CFAR
%       col                         current column, index starts from 1
%       N_c                         # of chirps
%       Nt                          # of Tx antennas
% output
%       X2_Detected                 updated input
%%
% In order to utilize the 'mod' function
index = mod(round((col:N_c/(Nt+1):col+Nt*N_c/(Nt+1))-1), ...
        length(X2_Detected));
index = index + 1;

X2_Detected(index) = 0;
end

