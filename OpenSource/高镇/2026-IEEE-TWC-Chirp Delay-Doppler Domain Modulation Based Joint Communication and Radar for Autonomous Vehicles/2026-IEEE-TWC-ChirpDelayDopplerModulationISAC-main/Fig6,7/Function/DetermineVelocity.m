function [flagVelocity, velocity_index] = DetermineVelocity(X2_Detected, col,...
                                        N_c, Nt, flag, fv)
% Determine the velocity with Nc divided into Nt+1 parts
% 
% input
%       X2_Detected                 0 or 1 after CFAR
%       col                         current column, index starts from 1
%       N_c                         # of chirps
%       Nt                          # of Tx antennas
%       fv                          previous velocity, starts from 0
%       flag                        Beacon frame or DDM frame
% output
%       flagVelocity                indicator
%                                   0: There exists velocity ambiguity
%                                   1: There exists velocity
%                                   2: There exists no velocity
%       velocity_index              index of velocity, index starts from 1

%%
% to improve robustness
bias = 2;

% In order to utilize the 'mod' function
index = mod(round((col:N_c/Nt:col+(Nt-1)*N_c/Nt)-1), ...
    length(X2_Detected));
index = index + 1;
Sum = sum(X2_Detected(index));


if strcmp(flag, 'Beacon')

    if Sum == Nt
        flagVelocity = 1;
        distance = mod(abs(fv+1+bias - index),N_c/Nt);
        [~,distance_index] = min(distance);
        velocity_index = index(distance_index);
    else
        flagVelocity = 0;
        velocity_index = inf;
    end

elseif strcmp(flag, 'DDM')
    flagVelocity = 0;
    velocity_index = inf;
    if Sum == Nt
        
        for i = 0:N_c/Nt-1
            index_i = mod((floor(fv+i :N_c/Nt: fv+i+(Nt-1)*N_c/Nt)-1), ...
            length(X2_Detected)) + 1; 
            if isempty(setdiff(index_i,index))
                flagVelocity = 1;
                velocity_index = floor(fv+i);
                break;
            end
        end
    end

else
    error('flag error!')
end

end

