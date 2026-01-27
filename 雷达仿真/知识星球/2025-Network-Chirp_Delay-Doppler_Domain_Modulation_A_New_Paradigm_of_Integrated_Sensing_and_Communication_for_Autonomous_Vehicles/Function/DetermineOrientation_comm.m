function TargetList_ort = DetermineOrientation_comm( ...
                        EachFrameEst_in, ...
                        TargetList)
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
%       TargetList_ort              the newest TargetList after adding
%                                   orientation
% TODO
%       the number of column of 'AllLabel_array' should be equal to
%       'AllLabel_array_j' adaptively
%%
TargetList_ort = TargetList;

AllLabel_cell_1 = {EachFrameEst_in(1).TargetList(:).label};
AllLabel_array_1 = squeeze(char(( string(AllLabel_cell_1).' )));
tmp = repmat('A_0', size(AllLabel_array_1,1), 1) ...
    == AllLabel_array_1;
index = find(all(tmp, 2));        
% Calculate orientation
delta_x = TargetList_ort.range_average * sind(TargetList_ort.azi_average)...
            - EachFrameEst_in(1).TargetList(index).range_average * sind(EachFrameEst_in(1).TargetList(index).azi_average);        
delta_y = TargetList_ort.range_average * cosd(TargetList_ort.azi_average)...
                - EachFrameEst_in(1).TargetList(index).range_average * cosd(EachFrameEst_in(1).TargetList(index).azi_average);
range_abs = sqrt(delta_y^2 + delta_x^2);
if range_abs > 2 % a threshold that used to determine whether the orientation can be calculated
    TargetList_ort.orientation = asind(delta_x...
        /range_abs);
%     TargetList_ort.orientation = 0;
    TargetList_ort.orientationVar = 'small';
else
    TargetList_ort.orientation = EachFrameEst_in(end).TargetList(index).orientation;
    TargetList_ort.orientationVar = 'large';   
end    

end