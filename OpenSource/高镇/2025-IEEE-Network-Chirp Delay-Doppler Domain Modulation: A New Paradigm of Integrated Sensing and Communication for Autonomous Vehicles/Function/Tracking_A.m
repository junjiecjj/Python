function KF_cur = Tracking_A(...
                Para,...
                period,...                
                KF_pre,...
                TargetList, ...
                KF_pre_all)
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

PreKFLabel_array = squeeze(char(( KF_pre.label(:))));

% Necessary definition
dt = Para.N_period*Para.N_c*Para.TcAll;
% state-transition matrix
F = [1 0 dt 0;...
     0 1 0 dt;...
     0 0 1 0;...
     0 0 0 1];

for i = 1:numel(TargetList)
% conduct EKF for each target in the current TargetList    
    tmp = repmat(CurTargetLabel_array(i,:), size(PreKFLabel_array,1), 1) ...
        == PreKFLabel_array;
    index = find(all(tmp, 2));
    if isempty(index)
%%
    %   There is no previous tracking object in KF_pre that matches current
    % TargetList.
    %   We need construct a new tracking.
        KF_cur.label{i} = CurTargetLabel_array(i,:);
%         if strcmp(TargetList(i).orientationVar, 'large')
%             KF_cur.var{i} = 200*eye(4);
%         else
            KF_cur.var{i} = 0.2*eye(4);
%         end        
        if abs(TargetList(i).azi_average - TargetList(i).orientation) < 85
            vx = (TargetList(i).velocity_average - (Para.N_c/2 + 1)) ...
                / cosd(TargetList(i).azi_average - TargetList(i).orientation) ...
                * sind(TargetList(i).orientation) ...
                /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
            vy = (TargetList(i).velocity_average - (Para.N_c/2 + 1)) ...
                / cosd(TargetList(i).azi_average - TargetList(i).orientation) ...
                * cosd(TargetList(i).orientation) ...
                /(2*Para.N_c*Para.TcAll*Para.fc/3e8);                
        else
            % the difference between azi and orientation is too big
            vx = rand;
            vy = 10*rand;
        end
    
        KF_cur.mu{i} = [TargetList(i).range_average * sind(TargetList(i).azi_average)...
                        /(4*Para.fs*Para.TcEff)...
                        *(3e8 * 2*(Para.Tsen+Para.Tcom)) ;... 
                        ...
                        TargetList(i).range_average * cosd(TargetList(i).azi_average)...
                        /(4*Para.fs*Para.TcEff)...
                        *(3e8 * 2*(Para.Tsen+Para.Tcom)) ;... 
                        ...
                        vx; ...
                        vy];     
       
    else
%%        
    %   There exists a previous tracking target that matches current
    % TargetList.
        KF_cur.label{i} = PreKFLabel_array(index,:);
        var_meas = 2;% prediction noise
        Cov_meas = var_meas*eye(3);% measurement noise
        FirstPeriod = DetermineFirstPeriod(KF_pre_all, KF_cur.label{i}, period, floor(Para.orientation_RefNum/2));
%         if (period - FirstPeriod) < (floor(Para.orientation_RefNum/2)-1) || strcmp(TargetList(i).orientationVar, 'large')
%             Cov_pred = 100*var_meas*eye(4);% prediction noise
%         else
%             Cov_pred = 0.01*var_meas*eye(4);% prediction noise
            Cov_pred = [0.01*var_meas 0 0 0;...
                        0 0.01*var_meas 0 0;...
                        0 0 0.03*var_meas 0 ;...
                        0 0 0 0.03*var_meas];
%         end                

        y = [ (TargetList(i).range_average - 1)...% don't forget minus 1
              /(4*Para.fs*Para.TcEff)...
              *(3e8 * 2*(Para.Tsen+Para.Tcom));...
              ...
              TargetList(i).azi_average;...
              ...
               (TargetList(i).velocity_average - (Para.N_c/2 + 1))...
              /(2*Para.N_c*Para.TcAll*Para.fc/3e8)];  

        % prediction
        mu_pred = F*cell2mat(KF_pre.mu(index));
        Var_pred = F*cell2mat(KF_pre.var(index))*F.' + Cov_pred;
        % paritial    
        C =  p_h_p_s(mu_pred, TargetList(i).orientation);
        % Kalman gain
        K = Var_pred*C.'/(C*Var_pred*C.' + Cov_meas);
        % posterior
        KF_cur.mu{i} = mu_pred + K*(y - h(mu_pred(1), mu_pred(2), mu_pred(3), mu_pred(4), TargetList(i).orientation));
        KF_cur.var{i} = (eye(4) - K*C)*Var_pred;        
    end
end
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