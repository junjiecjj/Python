function FirstPeriod = DetermineFirstPeriod(KF, label, period, NumKF)

FirstPeriod = period;
if size(KF,2) == 0
    return
end


for i = numel(KF):-1:1
    % if KF(i).label doesn't exist in KF4plot, then add it in
    AllLabel_i = char(KF(i).label);
    index = find (...
                all( repmat(label, size(AllLabel_i, 1), 1)...
                == AllLabel_i , 2)...
                );    
    if (~isempty(index)) 
        FirstPeriod = i+period-NumKF;        
    else
        return
    end
end
end