function TargetList = MoveTargetListWithAngleInfo(TargetList)
%% Move
% Premise: range and velocity are very close to each other
% don't add any target, only move internal data, i.e., move
% one date from one target to anoter target
% 
% input
%       TargetList          struct, used to save target information
% output
%       TargetList          updated TargetList
%%
if isempty(TargetList(1).range)
    NumTarget = 0;
else
    NumTarget = length(TargetList);
end

for i = 1:NumTarget
    NumData = length(TargetList(i).range);    
    for j = NumData:-1:1% In order to avoid error because NumData is not a constant in this loop
        index = inf;
        distance_azi = abs(TargetList(i).azi(j) - mean(TargetList(i).azi));                  
        distance_ele = abs(TargetList(i).ele(j) - mean(TargetList(i).ele));     
        % Determine the index of target which has the smallest average
        % distance with current data        
        for k = 1:NumTarget
            if k ~= i% not compare with self
                if abs(mean(TargetList(i).range) - mean(TargetList(k).range))<3 && ...% range and velocity are very close to each other
                        abs(mean(TargetList(i).velocity) - mean(TargetList(k).velocity))<3
                    if distance_azi > abs(TargetList(i).azi(j) - mean(TargetList(k).azi)) ||...
                       distance_ele > abs(TargetList(i).ele(j) - mean(TargetList(k).ele))
                        index = k;
                        distance_azi = abs(TargetList(i).azi(j) - mean(TargetList(k).azi));
                        distance_ele = abs(TargetList(i).ele(j) - mean(TargetList(k).ele));
                    end
                end
            end
        end
        % Move TargetList(i).angle(j) to TargetList(index).angle
        if index ~= inf
            % Add
            TargetList(index).range = [TargetList(index).range TargetList(i).range(j)];
            TargetList(index).velocity = [TargetList(index).velocity TargetList(i).velocity(j)];
            TargetList(index).azi = [TargetList(index).azi TargetList(i).azi(j)];
            TargetList(index).ele = [TargetList(index).ele TargetList(i).ele(j)];
            % Delete
            TargetList = DeleteTarget(TargetList, i, j);
        end
    end
end
end