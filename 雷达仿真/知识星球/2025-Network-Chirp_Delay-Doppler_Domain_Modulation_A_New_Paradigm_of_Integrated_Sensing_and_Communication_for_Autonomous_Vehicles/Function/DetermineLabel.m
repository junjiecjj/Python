function TargetList_cur = DetermineLabel(TargetList_pre,...
                            TargetList_cur, ...
                            Para)
%   Determine the label of the TargetList, in my simplified model to verify
% the feasibility of the Lodar, we only have two targets.
%   We use labeled TargetList_pre to determine the label of TargetList_cur.
% 
% input
%       TargetList_pre              TargetList with label of the last
%                                   period
%       TargetList_cur              current TargetList
%       Para                        struct of system parameters
% output
%       TargetList_cur              current TargetList
% 
% Deficiencies, and need to refine in the future version
%   1. finding the appropriate current target corresponding to the previous
% target through ' [~, index] = min(ParaAll_error_sum); ' is not optimal.
%   2. we assume communication receiver always exists
%%
% Determine the # of target in TargetList_cur
if isempty(TargetList_pre(1).range)
    NumTarget_pre = 0;
else
    NumTarget_pre = length(TargetList_pre);
end
if isempty(TargetList_cur(1).range)    
    error('NumTarget_cur = 0')
else
    NumTarget_cur = length(TargetList_cur);
end

% Determine the largest label of previous targets
AllLabel_pre_cell = {TargetList_pre(:).label};
AllLabel_pre_array = squeeze(char(( string(AllLabel_pre_cell).' )));
AllLabel_pre_num = zeros(NumTarget_pre, 1);
parfor i = 1:NumTarget_pre
    AllLabel_pre_num(i) = str2double(extractAfter(AllLabel_pre_array(i,:), '_'));
end
[LargestLabelValue, LargestLabelIndex] = max(AllLabel_pre_num);

% Determine the info of all the previous targets and current targets
ParaAllPre = zeros(3, NumTarget_pre);
parfor i = 1:NumTarget_pre
    ParaAllPre(:,i) = [TargetList_pre(i).range_average;...
                    TargetList_pre(i).velocity_average;...
                    TargetList_pre(i).azi_average];
end
ParaAllCur = zeros(3, NumTarget_cur);

% Previous targets matching existing targets 
parfor i = 1:NumTarget_cur
    TargetList_cur(i).range_average = mean(TargetList_cur(i).range);
    TargetList_cur(i).velocity_average = mean(TargetList_cur(i).velocity);
    TargetList_cur(i).azi_average = mean(TargetList_cur(i).azi);   
    ParaAllCur(:,i) = [TargetList_cur(i).range_average;...
                    TargetList_cur(i).velocity_average;...
                    TargetList_cur(i).azi_average];
end
for i = 1:NumTarget_pre
    ParaAll_error = abs(ParaAllCur - repmat(ParaAllPre(:,i), 1, NumTarget_cur));
    ParaAll_error_sum = sum(ParaAll_error);
    [~, index] = min(ParaAll_error_sum);% Can be refine, since this operation is not optimum
    % Determine whether it is reasonable or not
    if DetermineWhetherSameTarget(ParaAllPre(:, i), ...
                                ParaAllCur(:, index), ...
                                Para)
        TargetList_cur(index).label = TargetList_pre(i).label;
    end
end


% Give existing targets new label
flag = 1;
for i = 1:NumTarget_cur
    if isempty(TargetList_cur(i).label) == 1
        TargetList_cur(i).label = ['C_' num2str(flag+LargestLabelValue)];
        flag = flag + 1;
    end
end
end

function flag = DetermineWhetherSameTarget(ParaPre, ...
                                ParaCur, ...
                                Para)
%%
if      abs(ParaCur(1) - ParaPre(1)) <= Para.TcAll * Para.N_c * Para.N_period * Para.velocity_max / Para.distance_resolution ...
        &&...
        abs(ParaPre(2) - ParaCur(2)) <= 7 ...
        &&...
        abs(ParaPre(3) - ParaCur(3)) <= 9
    flag = 1;
else     
    flag = 0;
end
end