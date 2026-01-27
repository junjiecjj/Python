function TargetList = AppendTargetList(TargetList, range_index, velocity_index, Nf, Nc)
% Add the estimation results to the target list
% 
% input 
%       TargetList          struct, used to save target information
%       range_index         # of distance resolution, start from 1
%       velocity_index      # of velocity resolution, start from 1
%       Nf                  # of distance grids
%       Nc                  # of velocity grids
% output
%       TargetList          updated input

%%
if range_index <= Nf-3 && velocity_index <= Nc-2 &&...
        range_index >= 3 && velocity_index >= 2
% in orde to avoid spectrum leakage, e.g., from 1 to 256
    if isempty(TargetList(1).range)
        NumTarget = 0;
    else
        NumTarget = length(TargetList);
    end
    flag = 1;    
    Buff = struct('index',0,...
                  'range_distance', inf, ...
                  'velocity_distance', inf);
    for i = 1:NumTarget
        % determine range
        range_diff = range_index - TargetList(i).range;
        % determine velocity
        velocity_diff = velocity_index - TargetList(i).velocity;
    
        % decide whether to add or not
        if any(abs(range_diff) <= 5) && any(abs(velocity_diff) <= 3)
            flag = 0;% can be classified to current target
            if (mean(abs(range_diff)) + mean(abs(velocity_diff))) < ...
                    (Buff.range_distance + Buff.velocity_distance)
                Buff.index  = i;
                Buff.range_distance = mean(abs(range_diff));
                Buff.velocity_distance = mean(abs(velocity_diff));
            end
        end
    end
    if flag% Create a new target since range or velocity is very different from existing targets
        TargetList(NumTarget+1).range = range_index;
        TargetList(NumTarget+1).velocity = velocity_index;
    else% Select the existing target with smallest distance 
        TargetList(Buff.index).range = [TargetList(Buff.index).range range_index];
        TargetList(Buff.index).velocity = [TargetList(Buff.index).velocity velocity_index];
    end
else
    disp('Range or Velocity is either too small or too big!')
end
end

