function TargetList_ini = DetermineLabel_ini( ...
                    TargetList_ini, ...
                    Para, ...
                    flag)
%   Determine the label of the TargetList_ini by using Para since the
% interaction bewteen Radar Transciever and Comm Transciever.
% 
% input
%       TargetList_ini              initial TargetList
%       Para                        struct of system parameters
%       flag                        'radar' or 'comm' to indicate which receiver
% output
%       TargetList_ini              initial TargetList after adding label
%%
if strcmp(flag, 'radar')
% find target B in Para that is the most close to the target in
% TargetList_ini and named it 'B'
    NumTarget = length(TargetList_ini);
    ParaB = [Para.Target_A(1).fr;...
            Para.Target_A(1).fv + Para.N_c/2 + 1;...
            Para.Target_A(1).azi];
    ParaAll = zeros(3, NumTarget);
    for i = 1:NumTarget
        TargetList_ini(i).range_average = mean(TargetList_ini(i).range);
        TargetList_ini(i).velocity_average = mean(TargetList_ini(i).velocity);
        TargetList_ini(i).azi_average = mean(TargetList_ini(i).azi);
        ParaAll(:,i) = [TargetList_ini(i).range_average;...
                        TargetList_ini(i).velocity_average;...
                        TargetList_ini(i).azi_average];
    end
    ParaAll_error = abs(ParaAll - repmat(ParaB, 1, NumTarget));
    ParaAll_error_sum = sum(ParaAll_error);
    [~, index] = min(ParaAll_error_sum);
    
    % Assign label to TargetList
    for i = 1:NumTarget
        if i < index
            TargetList_ini(i).label = ['C_' num2str(i)];
        elseif i == index
            TargetList_ini(i).label = 'B_0';
        else
            TargetList_ini(i).label = ['C_' num2str(i-index)];
        end
    end
else% strcmp(flag, 'comm')
% find target A in Para that is the most close to the target in
% TargetList_ini and named it 'A'
    NumTarget = length(TargetList_ini);
    ParaA = [Para.Target_B(1).fr;...
            Para.Target_B(1).fv + Para.N_c/2 + 1;...
            Para.Target_B(1).azi];
    ParaAll = zeros(3, NumTarget);
    for i = 1:NumTarget
        TargetList_ini(i).range_average = mean(TargetList_ini(i).range);
        TargetList_ini(i).velocity_average = mean(TargetList_ini(i).velocity);
        TargetList_ini(i).azi_average = mean(TargetList_ini(i).azi);
        ParaAll(:,i) = [TargetList_ini(i).range_average;...
                        TargetList_ini(i).velocity_average;...
                        TargetList_ini(i).azi_average];
    end
    ParaAll_error = abs(ParaAll - repmat(ParaA, 1, NumTarget));
    ParaAll_error_sum = sum(ParaAll_error);
    [~, index] = min(ParaAll_error_sum);
    
    % Assign label to TargetList
    for i = 1:NumTarget
        if i < index
            TargetList_ini(i).label = ['C_' num2str(i)];
        elseif i == index
            TargetList_ini(i).label = 'A_0';
        else
            TargetList_ini(i).label = ['C_' num2str(i-index)];
        end
    end
end
end