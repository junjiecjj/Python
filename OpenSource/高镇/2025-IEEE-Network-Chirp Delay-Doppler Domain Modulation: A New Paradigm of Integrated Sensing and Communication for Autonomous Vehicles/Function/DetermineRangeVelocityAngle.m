function [TargetList] = DetermineRangeVelocityAngle(TargetList_old, X, ...
                        X2_Detected, Nt, Nr, N_c, angle_range, AntSpa,...
                        flag)
% Determine angle after range and velocity have been determined
% 
% input
%       TargetList_old              TargetList whose range and velocity have been determined
%       X                           RDM without any operation, Nf×Nc×Nr
%       X2_Detected                 0 or 1 after CFAR, Nf×Nc
%       Nt                          struct of Tx antennas
%       Nr                          struct of Rx antennas
%       N_c                         # of chirps
%       angle_range                 field of view
%       AntSpa                      struct of Antenna spacing
%                                       AntSpa.azi
%                                       AntSpa.ele
%       flag                        used to determine radar receiver and
%                                   comm receiver
% output
%       TargetList                  updated version of TargetList_old with
%                                   added angle

%% Find 1s' indices in X2_Detected
% first range then velocity, both start with the smaller indices
[row, col] = find(X2_Detected.');
tmp=row;
row=col;
col=tmp;
%% Determine range and velocity and classification
NumDetected = length(row);
TargetList = struct('range', [], ...
                    'velocity', [],...
                    'azi', [],...
                    'ele', [],...
                    ...
                    'range_average',[],...
                    'velocity_average', [],...
                    'azi_average', [],...
                    'ele_average', [],...
                    ...
                    'range_variance',[],...
                    'velocity_variance', [],...
                    'azi_variance',[], ...
                    'ele_variance',[]);
for i = 1:NumDetected
    flagVelocity = DetermineVelocityWithTargetList(TargetList_old, row(i), col(i));
    if flagVelocity == 0
%         disp('There exists no velocity!')                
    else
        % Fuction AppendTargetList() needs determine whether to add a new one
        % or just append to an old one
        TargetList = AppendTargetList(TargetList, row(i), col(i),...
                                     size(X2_Detected, 1), size(X2_Detected, 2));
    end
end
%% Determine angle
parfor i = 1:length(TargetList)
    for j = 1:length(TargetList(i).range)
%% |    Calculate phase from all antenna
        if strcmp(flag, 'radar')
            index_Nc = mod((TargetList(i).velocity(j) : N_c/(Nt.tot) : TargetList(i).velocity(j)+(Nt.tot-1)*N_c/(Nt.tot))-1,...
                    N_c)+1;
            PhaseAllAntenna = zeros(Nt.ele*Nr.ele, Nt.azi*Nr.azi);
            for i_R = 0:Nr.tot-1
                Phase = X(TargetList(i).range(j), index_Nc, i_R+1);
                for i_T = 0:Nt.tot-1
                    PhaseAllAntenna(floor(i_T/Nt.azi)*Nr.ele + floor(i_R/Nr.azi) + 1,...
                        mod(i_T, Nt.azi)*Nr.azi + mod(i_R, Nr.azi) + 1)...
                        = Phase(i_T+1);
                end
            end
        else% strcmp(flag, 'comm')
            index_Nc = mod((TargetList(i).velocity(j) : N_c/(Nt.tot) : TargetList(i).velocity(j)+(Nt.tot-1)*N_c/(Nt.tot))-1,...
                    N_c)+1;
            PhaseAllAntenna = zeros(Nr.azi*Nr.ele, Nt.azi*Nt.ele);
            for i_R = 0:Nr.tot-1
                Phase = X(TargetList(i).range(j), index_Nc, i_R+1);                
                PhaseAllAntenna(floor(i_R/Nr.azi)*Nr.azi + mod(i_R, Nr.azi) + 1,...
                    :)...
                    = Phase(:);                
            end
        end
%% |    With window and direct take the maximum after DFT       
        angle = ObtainAngleFromArray(PhaseAllAntenna, angle_range, AntSpa, flag, ...
                                    Nr, Nt);
%% |    Add the angle into the TargetList          
        TargetList(i).azi(j) = angle.azi;
        TargetList(i).ele(j) = angle.ele;
    end
end
%% Adjust TargetList according to the new angle information
TargetList = MoveTargetListWithAngleInfo(TargetList);
TargetList = CreatTargetListWithAngleInfo(TargetList, Nt, Nr, angle_range);
end

