function KF_cur = Tracking_B(...
                Para,...
                frame,...                                
                TargetList, ...
                KF_pre)
% EKF for conducting info fusion of prediction and measurement
% 
% input
%       Para                            system parameters
%       period                          No. period
%       KF_pre                          previous KF fusion results
%       TargetList                      TargetList of current period
% output
%       KF_cur                          current KF fusion results
% TODO
%       the number of column of 'CurTargetLabel_array(i,:)' should be equal to
%       'AllLabel_array_j' adaptively
%%
KF_cur = ...
        struct( ...
    'label',[],...
    'mu', [], ...
    'var', []);

% Obtain the label of TargetList
CurTargetLabel_cell = {TargetList(:).label};% cell array
CurTargetLabel_array = squeeze(char(( CurTargetLabel_cell )));% normal array

% Necessary definition
dt = Para.N_c*Para.TcAll;
% state-transition matrix
F = [1 0 dt 0;...
     0 1 0 dt;...
     0 0 1 0;...
     0 0 0 1];

tmp = repmat('A_0', size(CurTargetLabel_array,1), 1) ...
        == CurTargetLabel_array;
index = find(all(tmp, 2));

KF_cur.label = 'A_0';
var_meas = 2;% prediction noise
Cov_meas = var_meas*eye(3);% measurement noise
% if frame < Para.N_period*floor(Para.orientation_RefNum/2) || strcmp(TargetList(index).orientationVar, 'large')
%     Cov_pred = 0.1*var_meas*eye(4);% prediction noise
% else
%     tmp = 0.1/...
%         (1.1^(frame - Para.N_period*floor(Para.orientation_RefNum/2) + 1));
%     if tmp >= 0.001
%         Cov_pred = tmp*var_meas*eye(4);
%     else
%         Cov_pred = 0.001*var_meas*eye(4);
%     end
% end    
Cov_pred = 0.001*var_meas*eye(4);

y = [ (TargetList(index).range_average - 1)...% don't forget minus 1
      /(2*Para.fs*Para.TcEff)...
      *(3e8 * 2*(Para.Tsen+Para.Tcom));...
      ...
      TargetList(index).azi_average;...
      ...
       (TargetList(index).velocity_average - (Para.N_c/2 + 1))...
      /(Para.N_c*Para.TcAll*Para.fc/3e8)];  

% prediction
mu_pred = F*KF_pre.mu;
Var_pred = F*KF_pre.var*F.' + Cov_pred;
% paritial    
C =  p_h_p_s(mu_pred, TargetList(index).orientation);
% Kalman gain
K = Var_pred*C.'/(C*Var_pred*C.' + Cov_meas);
% posterior
KF_cur.mu = mu_pred + K*(y - h(mu_pred(1), mu_pred(2), mu_pred(3), mu_pred(4), TargetList(index).orientation));
KF_cur.var = (eye(4) - K*C)*Var_pred;   

end
%% Measurement Fuction
function h_x = h(x, y, vx, vy, o)
    h_x = zeros(3,1);
    h_x(1) = sqrt(x^2 + y^2);
    h_x(2) = asind(x/h_x(1));
    if o < 45
        h_x(3) = vy*cosd(h_x(2) - o)/cosd(o);
    else
        h_x(3) = vx*cosd(h_x(2) - o)/sind(o);
    end
end

%% partial derivative of the function with respect to the state
function C = p_h_p_s(mu_pred, o)
    theta = asind(mu_pred(1)/sqrt(mu_pred(1)^2 + mu_pred(2)^2));
    C = zeros(3, 4);          
    C(1, 1) = mu_pred(1)/sqrt(mu_pred(1)^2 + mu_pred(2)^2); % pr_px
    C(1, 2) = mu_pred(2)/sqrt(mu_pred(1)^2 + mu_pred(2)^2); % pr_py
    C(2, 1) = abs(mu_pred(2))/ ( ...
        mu_pred(2)^2 + mu_pred(1)^2)...
                * 180/pi;% ptheta_px
    C(2, 2) = (-mu_pred(2)*mu_pred(1))/ (abs(mu_pred(2))*(mu_pred(2)^2 + mu_pred(1)^2))...
                * 180/pi;% ptheta_py  
    if abs(o) < 45        
        C(3, 1) = -mu_pred(4)*sind(theta-o)/cosd(o) * abs(mu_pred(2))/sqrt(mu_pred(1)^2 + mu_pred(2)^2) ;
        C(3, 2) = mu_pred(4)*sind(theta-o)/cosd(o) * mu_pred(1)*mu_pred(2)/(sqrt(mu_pred(1)^2 + mu_pred(2)^2) * abs(mu_pred(2))) ;
        C(3, 4) = cosd(asind(mu_pred(1)/sqrt(mu_pred(1)^2+mu_pred(2)^2))-o) / cosd(o);% pvr_pvy
    else
        C(3, 1) = -mu_pred(4)*sind(theta-o)/sind(o) * abs(mu_pred(2))/sqrt(mu_pred(1)^2 + mu_pred(2)^2) ;
        C(3, 2) = mu_pred(4)*sind(theta-o)/sind(o) * mu_pred(1)*mu_pred(2)/(sqrt(mu_pred(1)^2 + mu_pred(2)^2) * abs(mu_pred(2))) ;
        C(3, 3) = cosd(asind(mu_pred(1)/sqrt(mu_pred(1)^2+mu_pred(2)^2))-o) / sind(o);% pvr_pvx        
    end
end