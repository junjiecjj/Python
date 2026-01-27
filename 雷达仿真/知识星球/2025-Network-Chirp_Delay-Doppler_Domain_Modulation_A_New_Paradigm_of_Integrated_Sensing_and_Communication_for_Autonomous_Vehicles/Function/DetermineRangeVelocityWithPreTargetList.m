function TargetList = DetermineRangeVelocityWithPreTargetList(TargetList_pre, ...
            Para, ...
            X2_Detected, ...
            i_frame)
%%
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
if mod(i_frame , Para.N_period) ~= 9
    % find minimum distance
    [row, col] = find(X2_Detected.');
    tmp=row;
    row=col;
    col=tmp;
    min_distance = row(1);
    count = 1;
    for i = 1:length(row)
        if i == length(row)
            break;
        elseif row(i+1) - row(i) < 3         
            count = count + 1;
            min_distance = min_distance + row(i+1);
        else
            break;
        end
    end
    min_distance = min_distance / count;
    
    % determine the index of A_0
    AllLabel_cell = {TargetList_pre(:).label};
    AllLabel_array = squeeze(char(( AllLabel_cell )));
    tmp = repmat('A_0', size(AllLabel_array,1), 1) ...
        == AllLabel_array;
    index = find(all(tmp, 2));
    
    % determine the start point
    rangeStart = ceil(min_distance);
    % 
    velocityStart = mod(round(TargetList_pre(index).velocity_average)-1, Para.N_c)+1;
    FindVelStart = 1;%(i_frame>Para.N_period*Para.N_period_VelStartTxInfo);
    flag = 1;% to make the velocityStart more correct
    
    % determine range of range and velcoty
    direction_range = 1;
    
    range_range = [-1 0];% first is positive direction    
    for j = 1:2% two directions, this loop is for range
        while true
            range_range(j) = range_range(j) + direction_range;
            velocity_range = [-1 0];
            direction_velocity = 1;
            for i = 1:2% two directions, this loop is for velocity
                while true
                    velocity_range(i) = velocity_range(i) + direction_velocity;
                    velocity = mod(velocityStart + velocity_range(i) - 1 + (0:Para.N_c/Para.Nt.tot:Para.N_c-Para.N_c/Para.Nt.tot), ...
                                   Para.N_c) + 1;
                    while FindVelStart% find the start velocity first, then determine the exact velocity
                        if sum(X2_Detected(range_range(j)+rangeStart, velocity)) < Para.Nt.tot
%                         if X2_Detected(range_range(j)+rangeStart, velocity) ~= 1
                            velocityStart = mod(velocityStart + 1 - 1, Para.N_c) + 1;
                            velocity = mod(velocityStart + velocity_range(i) - 1 + (0:Para.N_c/Para.Nt.tot:Para.N_c-Para.N_c/Para.Nt.tot), ...
                                   Para.N_c) + 1;
                        else                            
                            FindVelStart = 0;
                            velocity = mod(velocityStart + velocity_range(i) - 1 + (0:Para.N_c/Para.Nt.tot:Para.N_c-Para.N_c/Para.Nt.tot), ...
                                   Para.N_c) + 1;
                            break;
                        end
                    end
                    if sum(X2_Detected(range_range(j)+rangeStart, velocity)) == Para.Nt.tot
                        TargetList.range = [TargetList.range range_range(j)+rangeStart];
                        TargetList.velocity = [TargetList.velocity velocity(1)];        
                    else
                        direction_velocity = -1;
                        break
                    end                          
                end
            end
            if flag% to make the velocityStart more correct
                flag = 0;
                velocityStart = round(mean(TargetList.velocity));
            end
            if i == 2 && X2_Detected(range_range(j) + direction_range + rangeStart, velocityStart) ~= 1
                direction_range = -1;
                break
            end
        end
    end
    TargetList.range_average = mean(TargetList.range);
    TargetList.velocity_average = mean(TargetList.velocity);
%%
else% mod(i_frame , Para.N_period) == 9
    TargetList_All = DetermineRangeVelocity(X2_Detected, Para.N_c, Para.Nt.tot);
    All_range_average = {TargetList_All(:).range_average};
    min_range = min(All_range_average{:});
    for i = 1:length(All_range_average)
        if TargetList_All(i).range_average == min_range        
            TargetList = TargetList_All(i);
            break
        end
    end
    TargetList.velocity = TargetList.velocity;
    TargetList.velocity_average = TargetList.velocity_average;
end
end