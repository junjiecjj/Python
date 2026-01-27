function TargetList = DetermineRangeVelocity(X2_Detected, N_c, Nt)
% Determine the range and velocity in X2_Detected,
% then Classify range, velocity according to spectrum leakage
%
% input
%       X2_Detected                 0 or 1 after CFAR
%       N_c                         # of chirps
%       Nt                          # of Tx antennas
% output
%       TargetList                  struct, used to save target information

%% Find 1s' indices in X2_Detected
% first range then velocity, both start with the smaller indices
X2_Detected_Copy = X2_Detected;% Used for set zero after detected
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
%     if col(i) < ceil(N_c/(Nt+1))% if not, there is no need to determine subsequent columns(velocities)
%     if col(i) < ceil(2*N_c/(Nt+1))% if not, there is no need to determine subsequent columns(velocities)
        if X2_Detected_Copy(row(i), col(i)) == 1% because this grid can be set to zero in the later step
            [flagVelocity, velocity_index] = DetermineVelocity(X2_Detected_Copy(row(i),:), col(i),...
                                        N_c, Nt);
            if flagVelocity == 0
%                 error(['There exists velocity ambiguity!', newline,...
%                        'This part will be completed in future version!'])                
            elseif flagVelocity == 2
%                 disp('There exists no velocity!')
            else
                % Fuction AppendTargetList() needs determine whether to add a new one
                % or just append to an old one
                TargetList = AppendTargetList(TargetList, row(i), velocity_index,...
                                             size(X2_Detected, 1), size(X2_Detected, 2));                
            end
%             X2_Detected_Copy(row(i),:) = RDMsetZero(X2_Detected_Copy(row(i),:), col(i), N_c, Nt);
        end
%     end
end
% if the minimum range/velocity of one target is between the minimum and
% maximum value of another target, then merge them
TargetList = MergeTargetList(TargetList);
for i = 1:length(TargetList)
    TargetList(i).range_average = mean(TargetList(i).range);
    TargetList(i).velocity_average = mean(TargetList(i).velocity);
end
end
%%
function TargetList = MergeTargetList(TargetList)
% if the minimum range/velocity of one target is between the minimum and
% maximum value of another target, then merge them
    if isempty(TargetList(1).range)
        NumTarget = 0;
    else
        NumTarget = length(TargetList);
    end
    for i = NumTarget:-1:1
        for j = i-1:-1:1
            if ((min(TargetList(i).range) > min(TargetList(j).range) && ...
               min(TargetList(i).range) < max(TargetList(j).range)) || ...
               (max(TargetList(i).range) > min(TargetList(j).range) && ...
               max(TargetList(i).range) < max(TargetList(j).range)))...
               && ...
               ((min(TargetList(i).velocity) > min(TargetList(j).velocity) && ...
               min(TargetList(i).velocity) < max(TargetList(j).velocity))|| ...
               (max(TargetList(i).velocity) > min(TargetList(j).velocity) && ...
               max(TargetList(i).velocity) < max(TargetList(j).velocity)))

                % Merge i-th target into j-th target                
                TargetList(j).range = [TargetList(j).range TargetList(i).range];
                TargetList(j).velocity = [TargetList(j).velocity TargetList(i).velocity];
                % release i-th Target
                TargetList(i) = [];
            end
            break;
        end
    end
end