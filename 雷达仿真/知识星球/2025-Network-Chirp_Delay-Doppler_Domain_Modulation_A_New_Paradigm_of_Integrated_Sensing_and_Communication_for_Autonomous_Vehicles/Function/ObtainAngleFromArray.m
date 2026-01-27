function angle = ObtainAngleFromArray(PhaseAllAntenna, angle_range, AntSpa ...
                                      , flag, Nr, Nt)
% Estimate angle from the phase of the array
% 
% input
%       PhaseAllAntenna         phase of the array, Nele × Nazi(flag == 'radar')
%                               Nr.ele*Nr.azi × Nt.ele*Nt.azi(flag == 'comm')
%       angle_range             field of view
%       AntSpa                  struct of Antenna spacing
%                               AntSpa.azi
%                               AntSpa.ele
%       flag                    used to determine radar receiver and
%                               comm receiver
%       Nr                      # of receiver antennas, Nr.azi, Nr.ele
%                               Nt.azi, Nt.ele
% output
%       angle                   struct, 1 × 2, azimuth and elevation

%%
azi = angle_range.azi(1):angle_range.azi(2);
ele = angle_range.ele(1):angle_range.ele(2);
if strcmp(flag, 'radar')
    [Nele, Nazi] = size(PhaseAllAntenna);
    % Generate coarse angle
    A = GenSteeringVec(azi, ele, Nazi, Nele, AntSpa, flag);
    a = reshape(PhaseAllAntenna, [Nazi*Nele, 1]);
    % a_win = a.*hanning(Nazi*Nele);
    result = a'*A;
    [~, index] = max(abs(result));
    angle.azi = angle_range.azi(1)+ mod(index-1, length(azi));
    angle.ele = angle_range.ele(1)+ floor((index-1)/length(azi));
else% strcmp(flag, 'comm')
    Ar = GenSteeringVec(azi, ele, Nr.azi, Nr.ele, AntSpa, flag);
    Rxx = PhaseAllAntenna*PhaseAllAntenna';
    [V, ~] = eig(Rxx);
    V_noise = V(:,1:end-1);
    tmp = sum(abs(Ar'*V_noise).^2, 2);
    [~, index] = min(tmp);        
    angle.azi = angle_range.azi(1)+ mod(index-1, length(azi));
    angle.ele = angle_range.ele(1)+ floor((index-1)/length(azi));    
end
end



















