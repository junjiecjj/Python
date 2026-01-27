function TargetList_ort = DetermineOrientation( ...
                        EachPeriodEst_in, ...
                        Para, ...
                        flag)
%   EachPeriodEst is a struct arry, whose each element is a TargetList.
%   Its index indicates the number of the Para.N_period.
%   In order to save communication overhead of MATLAB, we only pass previous
% Para.orientation_RefNum Para.N_period  TargetList to calculate the orientation. 
% 
%   We also give the possiable error of the orientation by calculating the
% absolute distance of motion. If absolute distance of motion is larger
% than 3m, we consider the error of the orientation is small.
% 
% input
%       EachPeriodEst_in             TargetList of the last 5 periods if  
%                                   length(ParaAll_A.EachPeriodEst) >=5, or
%                                   else TargetList of all the past periods
%       Para                        struct of system parameters
% output
%       EachPeriodEst_out            the newest TargetList after adding
%                                   orientation
% TODO
%       the number of column of 'AllLabel_array' should be equal to
%       'AllLabel_array_j' adaptively
%%
TargetList_ort = EachPeriodEst_in(end).TargetList;

NumPeriod = length(EachPeriodEst_in);
NumTarget = length(EachPeriodEst_in(end).TargetList);
% obtain all the label of the newest TargetList
AllLabel_cell = {EachPeriodEst_in(end).TargetList(:).label};
AllLabel_array = squeeze(char(( AllLabel_cell )));

for i = 1:NumTarget
    % find the oldest period that has label 'AllLabel_array(i,:)'
    for j = 1:NumPeriod
        AllLabel_cell_j = {EachPeriodEst_in(j).TargetList(:).label};
        AllLabel_array_j = squeeze(char(( string(AllLabel_cell_j).' )));
        tmp = repmat(AllLabel_array(i,:), size(AllLabel_array_j,1), 1) ...
            == AllLabel_array_j;
        index = find(all(tmp, 2));        
        if j ~= NumPeriod && isempty(index)            
            continue;
        elseif j == NumPeriod
            % if true, this target appears for the first time
            TargetList_ort(i).orientation = 0;
            TargetList_ort(i).orientationVar = 'large';
        else
        % Calculate orientation
            delta_x = TargetList_ort(i).range_average * sind(TargetList_ort(i).azi_average)...
                        - EachPeriodEst_in(j).TargetList(index).range_average * sind(EachPeriodEst_in(j).TargetList(index).azi_average);        
            delta_y = TargetList_ort(i).range_average * cosd(TargetList_ort(i).azi_average)...
                            - EachPeriodEst_in(j).TargetList(index).range_average * cosd(EachPeriodEst_in(j).TargetList(index).azi_average);
            range_abs = sqrt(delta_y^2 + delta_x^2);
            if range_abs > 2 % a threshold that used to determine whether the orientation can be calculated
                TargetList_ort(i).orientation = asind(delta_x...
                    /range_abs);
                TargetList_ort(i).orientationVar = 'small';
            else
                TargetList_ort(i).orientation = EachPeriodEst_in(end-1).TargetList(index).orientation;
                TargetList_ort(i).orientationVar = 'large';   
            end
            break;
        end
    end
end
end