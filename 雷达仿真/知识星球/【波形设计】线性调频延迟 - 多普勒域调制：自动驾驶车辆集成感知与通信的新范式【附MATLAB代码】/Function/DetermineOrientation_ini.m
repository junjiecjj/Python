function TargetList_ort = DetermineOrientation_ini( ...
                TargetList_ini, ...
                TargetList_ort, ...
                Para, ...
                flag)
%   Determine the orientation of TargetList_ort in the initialization
% process by comparing TargetList_ini and TargetList_ort.
% 
%   We also give the possiable error of the orientation by calculating the
% absolute distance of motion. If absolute distance of motion is larger
% than 3m, we consider the error of the orientation is small.
%   
% input
%       TargetList_ini              the first TargetList
%       TargetList_ort              the second TargetList, which is used to
%                                   compare with TargetList_ini to obatain
%                                   orientation
%       Para                        struct of system parameters
%       flag                        'radar' or 'comm' to indicate which receiver
% output
%       TargetList_ort              the second TargetList that has been
%                                   added orientation
% TODO
%       the number of column of 'TargetList_ort(i).label' should be equal
%       to 'LabelAll_ini' adaptively
%% 

if isempty(TargetList_ini(1).range)
    NumTarget_ini = 0;
else
    NumTarget_ini = length(TargetList_ini);
end
if isempty(TargetList_ort(1).range)
    NumTarget_ort = 0;
else
    NumTarget_ort = length(TargetList_ort);
end

% NumTarget_ini = length(TargetList_ini);
% NumTarget_ort = length(TargetList_ort);
LabelAll_ini = squeeze(char( {TargetList_ini(:).label} ));% str2mat( string({TargetList_ini(:).label}));
for i = 1:NumTarget_ort    
    % find the index of previous target that mathches i-th current target
    comparison = repmat(TargetList_ort(i).label, NumTarget_ini, 1) == LabelAll_ini;
    result = all(comparison, 2);
    [row, ~] = find(result);
    if isempty(row) == 0% if a target exists in both TargetList_ini and TargetList_ort
        % Calculate orientation
        delta_x = TargetList_ort(i).range_average * sind(TargetList_ort(i).azi_average)...
                    - TargetList_ini(row).range_average * sind(TargetList_ini(row).azi_average);        
        delta_y = TargetList_ort(i).range_average * cosd(TargetList_ort(i).azi_average)...
                        - TargetList_ini(row).range_average * cosd(TargetList_ini(row).azi_average);
        range_abs = sqrt((delta_y)^2  + delta_x^2);
        if range_abs > 2 % a threshold that used to determine whether the orientation can be calculated
            TargetList_ort(i).orientation = asind(delta_x...
                /range_abs);
            TargetList_ort(i).orientationVar = 'small';
        else
            TargetList_ort(i).orientation = 0;
            TargetList_ort(i).orientationVar = 'large';
        end
    else% if a target is new in TargetList_ort
        TargetList_ort(i).orientation = 0;
        TargetList_ort(i).orientationVar = 'large';
    end
    TargetList_ort(i).azi_average = mean(TargetList_ort(i).azi);
end

end