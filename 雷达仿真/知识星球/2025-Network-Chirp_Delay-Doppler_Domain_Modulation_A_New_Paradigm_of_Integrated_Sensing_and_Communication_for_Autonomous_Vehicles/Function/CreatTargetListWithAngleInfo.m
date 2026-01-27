function TargetList = CreatTargetListWithAngleInfo(TargetList, Nt, Nr, angle_range)
%% Creat
% Premise: range and velocity are very close to each other
% Creat new target according to angle information
% 
% input
%       TargetList          struct, used to save target information
%       Nt                  struct of Tx antennas
%       Nr                  struct of Rx antennas
% output
%       TargetList          updated TargetList
%%
while true
    if isempty(TargetList(1).range)
        NumTarget = 0;
    else
        NumTarget = length(TargetList);
    end
    TargetList_New = TargetList;
    for i = 1:NumTarget
        % More than or equal to 1 resolution
        % ☆☆☆This determine statement is very important!!!
        while max(TargetList(i).azi) - min(TargetList(i).azi) >= (angle_range.azi(2)-angle_range.azi(1))/(Nt.azi*Nr.azi) ||...
            max(TargetList(i).ele) - min(TargetList(i).ele) >= (angle_range.ele(2)-angle_range.ele(1))/(Nt.ele*Nr.ele)
            [~, index] = max(TargetList(i).azi);
            % Creat
            TargetList = CreatTarget(TargetList, i, index);
            % Delete
            TargetList = DeleteTarget(TargetList, i, index);
            % Update
            TargetList = MoveTargetListWithAngleInfo(TargetList);
            break;
        end   
        TargetList_New = TargetList;
    end
    if isequal(TargetList_New, TargetList)
        break;
    else
        TargetList = TargetList_New;
    end
end
end

