function [flagVelocity, velocity_index] = DetermineVelocity(X2_Detected, col,...
                                        N_c, Nt)
% Determine the velocity with Nc divided into Nt+1 parts
% 
% input
%       X2_Detected                 0 or 1 after CFAR
%       col                         current column, index starts from 1
%       N_c                         # of chirps
%       Nt                          # of Tx antennas
% output
%       flagVelocity                indicator
%                                   0: There exists velocity ambiguity
%                                   1: There exists velocity
%                                   2: There exists no velocity
%       velocity_index              index of velocity, index starts from 1

%%
% In order to utilize the 'mod' function
index = mod(round((col:N_c/(Nt+1):col+Nt*N_c/(Nt+1))-1), ...
        length(X2_Detected));
index = index + 1;

% Generate all 1 row vector
All_1 = ones(1, Nt+1);

Detected = zeros(1, Nt+1);
indicator = zeros(1, Nt+1);
for i = 1:Nt+1    
    if X2_Detected(index(i)) == 1
        Detected(i) = index(i);
        indicator(i) = 1;
    elseif X2_Detected(mod(index(i)-1 - 1, length(X2_Detected)) + 1) == 1
        Detected(i) = mod(index(i)-1 - 1, length(X2_Detected)) + 1;
        indicator(i) = 1;
    elseif X2_Detected(mod(index(i)+1 - 1, length(X2_Detected)) + 1) == 1
        Detected(i) = mod(index(i)+1 - 1, length(X2_Detected)) + 1;
        indicator(i) = 1;
    end    
end
% indicator = xor(All_1, X2_Detected(index));
indicator = xor(All_1, indicator);

SumOfXor = sum(indicator);
index_velocity = find(indicator);
if SumOfXor == 0
    flagVelocity = 0;
    velocity_index = inf;
elseif SumOfXor >= 2% previous version:2, in order to increase robustness
    flagVelocity = 2;
    velocity_index = inf;
else% SumOfXor == 1
    flagVelocity = 1;
%     velocity_index = index(index_velocity);
    velocity_index = index(index_velocity);
end

end

