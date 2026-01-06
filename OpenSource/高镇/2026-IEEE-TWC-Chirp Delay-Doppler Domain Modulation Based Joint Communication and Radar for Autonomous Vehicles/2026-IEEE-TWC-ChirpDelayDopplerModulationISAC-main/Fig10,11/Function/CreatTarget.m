function TargetList = CreatTarget(TargetList, i, j)
% Creat a new target at the end of the TargetList
% the information of the new target comes from the old j-th element of 
% i-th target
% 
% input
%       TargetList          struct, used to save target information
%       i                   i-th target
%       j                   j-th element
% output
%       TargetList          updated input       
%%
if isempty(TargetList(1).range)
    NumTarget = 0;
else
    NumTarget = length(TargetList);
end
TargetList(NumTarget+1).range = TargetList(i).range(j);
TargetList(NumTarget+1).velocity = TargetList(i).velocity(j);
TargetList(NumTarget+1).azi = TargetList(i).azi(j);
TargetList(NumTarget+1).ele = TargetList(i).ele(j);
end

