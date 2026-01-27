function TargetList = DeleteTarget(TargetList, i, index)
% Delete the index-th element of i-th target 
% 
% input
%       TargetList          struct, used to save target information
%       i                   i-th target
%       index               index-th element
% output
%       TargetList          updated input    
%%
TargetList(i).range = [TargetList(i).range(1:index-1) ...
                       TargetList(i).range(index+1:end)];
TargetList(i).velocity = [TargetList(i).velocity(1:index-1) ...
                          TargetList(i).velocity(index+1:end)];
TargetList(i).azi = [TargetList(i).azi(1:index-1) ...
                       TargetList(i).azi(index+1:end)];  
TargetList(i).ele = [TargetList(i).ele(1:index-1) ...
                       TargetList(i).ele(index+1:end)];  
end

