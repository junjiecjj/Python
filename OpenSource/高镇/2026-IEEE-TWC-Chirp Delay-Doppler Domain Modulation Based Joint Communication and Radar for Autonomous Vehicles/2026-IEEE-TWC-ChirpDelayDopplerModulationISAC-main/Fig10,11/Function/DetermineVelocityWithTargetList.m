function flagVelocity = DetermineVelocityWithTargetList(TargetList_Old, range_index, velocity_index)
%   We have had a TargetList by divide Nc into Nt+1 parts, now we re-transmit
% signal with Nc divided into Nt parts. 
%   Utilizing the TargetList created by the former RDM with Nc
% divided into Nt+1 parts, we want to determine the index we are now
% examining now whether exists a target.
% 
% input
%       TargetList_Old              old TargetList with Nc divided into Nt
%                                   parts
%       range_index                 the range index we are now examining
%       velocity_index              the velocity index we are now examining
% output
%       flagVelocity                indicator to show whether there exists
%                                   a target.
%                                   0: no
%                                   1: yes
%% 
NumTarget = length(TargetList_Old);
flagVelocity = 0;
for i = 1:NumTarget
    % determine range
    range_diff = range_index - TargetList_Old(i).range;
    % determine velocity
    velocity_diff = velocity_index - TargetList_Old(i).velocity;

    % decide whether to add or not
    if sum((abs(range_diff) <= 1) & (abs(velocity_diff) <= 1)) >= 1
        flagVelocity = 1;
        break;
    end 
end
end

