function [TargetList, ...
          QAM_est, ...
          IM_rng_est, ...
          IM_vel_est, ...
          Gain_RDM_cur,...
          range_average,...
          velocity_average] = ... 
                         DetermineAll_A( ...
                         i_frame, ...
                         X2_Detected, ...
                         Para, ...
                         X, ...
                         TDData_B,...
                         KF, ...
                         Gain_RDM_pre, ...
                         TargetList_pre_All,...
                         range_average_pre,...
                         velocity_average_pre)
% Determine all information, including bit, range, velocity, angle of
% arrival, label and orientation
%% |    Preliminary
predict_x = KF.mu(1) + KF.mu(3)*Para.TcAll*Para.N_c;
predict_y = KF.mu(2) + KF.mu(4)*Para.TcAll*Para.N_c;
predict_azi = atand(predict_x/predict_y);
predict_r = sqrt(predict_x^2 + predict_y^2);
predict_relative_r = sqrt((KF.mu(3)*Para.TcAll*Para.N_c)^2 + (KF.mu(4)*Para.TcAll*Para.N_c)^2)*(2*Para.fs*Para.TcEff/...
                (3e8 * 2*(Para.Tsen+Para.Tcom))) ;


if i_frame == 1
    TargetList_pre = TargetList_pre_All;
else    
    TargetList_pre = TargetList_pre_All(end).TargetList;
end
TargetList = DetermineRangeVelocityWithPreTargetList(...
    TargetList_pre, ...
    Para, ...
    X2_Detected, ...
    i_frame);
min_distance = TargetList.range_average;
est_velocity = TargetList.velocity_average;

% obtain all the label of the previous TargetList
AllLabel_cell = {TargetList_pre(:).label};
AllLabel_array = squeeze(char(( AllLabel_cell )));
tmp = repmat('A_0', size(AllLabel_array,1), 1) ...
    == AllLabel_array;
index = find(all(tmp, 2)); 
predict_azi_tx = -(predict_azi-TargetList_pre(index).orientation);
%% |    Determine Bit
% azi = asind(KF.mu(1) / sqrt(KF.mu(1)^2 + KF.mu(2)^2));
range_average = sqrt(...
                (KF.mu(1) + KF.mu(3)*Para.TcAll*Para.N_c)^2 ...
                + (KF.mu(2) + KF.mu(4)*Para.TcAll*Para.N_c)^2)...
                *(2*Para.fs*Para.TcEff/...
                (3e8 * 2*(Para.Tsen+Para.Tcom))) + 1;% don't forget plus 1

% range_average = TargetList_pre(index).range_average ...
%     + (TargetList_pre(index).velocity_average - Para.N_c/2 - 1) ...
%     /(Para.N_c*Para.TcAll*Para.fc/3e8)...
%     /(2*Para.fs*Para.TcEff ...
%     /(3e8 * 2*(Para.Tsen+Para.Tcom)))...
%     *Para.N_c*Para.TcAll;

% velocity_average = (KF.mu(3)*sind(predict_azi) + ...
%                    KF.mu(4)*cosd(predict_azi))...
%                    *Para.N_c*Para.TcAll*Para.fc/3e8+ Para.N_c/2 + 1;
if i_frame > Para.N_period*Para.N_period_VelStartTxInfo
    velocity_average = (velocity_average_pre(2) + velocity_average_pre(1))/2;
else
    velocity_average = TargetList_pre.velocity_average;
end

% Determine IM
IM_rng_est = round(min_distance - range_average);% don't forget to minus 1 since IM_rng start from 0
fprintf('real range                     = %4.2f\n', Distance(Para.CarCenter(1,:), Para.CarCenter(2,:)) * 2*Para.fs*Para.TcEff/...
            (3e8 * 2*(Para.Tsen+Para.Tcom)) + 1)
fprintf('range_average                  = %4.2f\n', range_average)                   
fprintf('min_distance                   = %4.2f\n', min_distance)
fprintf('min_distance - range_average   = %4.2f\n', min_distance - range_average)
fprintf('IM_rng_est                     = %4.2f<<<<<<\n', IM_rng_est)
disp('||||||||||||||||||||||||||||||||||')


% % In order to avoid the error when i_fram = 9+10n
% if mod(i_frame, Para.N_period) == Para.N_period-1
%     IM_vel_est = floor(est_velocity - velocity_average);
% else
%     IM_vel_est = round(est_velocity - velocity_average);
% end
fprintf('real velocity                  = %4.2f\n', (Para.velocity(2) - Para.velocity(1))*...
            (Para.CarCenter(2,2) - Para.CarCenter(1,2))/...
            Distance(Para.CarCenter(1,:), Para.CarCenter(2,:)) ...
            *Para.N_c*Para.TcAll*Para.fc/3e8 + Para.N_c/2 + 1)
fprintf('velocity_average               = %4.2f\n', velocity_average)
fprintf('est_velocity                   = %4.2f\n', est_velocity)
fprintf('est_velocity - velocity_average= %4.2f\n', est_velocity - velocity_average)
% if abs(velocity_average - est_velocity) >= 0.5 && i_frame >= Para.N_period*Para.N_period_VelStartTxInfo
%     disp('pause')
% end

IM_vel_est = mod(round(est_velocity - velocity_average),Para.N_c/Para.Nt.tot);% don't forget to minus 1 since IM_vel start from 0
if round(est_velocity - velocity_average)<0
    IM_vel_flag = 1;
else
    IM_vel_flag = 0;
end
fprintf('IM_vel_est                     = %4.2f<<<<<<\n', IM_vel_est)
disp('||||||||||||||||||||||||||||||||||')

if IM_rng_est < 0
    IM_rng_est = 0;
elseif IM_rng_est > Para.N_f/2 - 1
    IM_rng_est = Para.N_f/2 - 1;
end



% if IM_vel_est < 0
%     IM_vel_est = mod(round(IM_vel_est), Para.N_c/Para.Nt.tot);
% %     IM_vel_est = 0;
% elseif IM_vel_est > Para.N_c/Para.Nt.tot -1
%     if IM_vel_est >= Para.N_c/Para.Nt.tot - 0.5
%         IM_vel_est = 0;
%     else
%         IM_vel_est = Para.N_c/Para.Nt.tot -1;
%     end
% %     IM_vel_est = Para.N_c/Para.Nt.tot -1;
% end
if i_frame <= Para.N_period*Para.N_period_VelStartTxInfo
    IM_vel_est = 0;
end

% final result
TargetList.range = (TargetList.range - IM_rng_est);
if i_frame > Para.N_period*Para.N_period_VelStartTxInfo
    TargetList.velocity = (TargetList.velocity - IM_vel_est + IM_vel_flag*Para.N_c/Para.Nt.tot);
    % while mean(TargetList.velocity) - mean(TargetList_pre(index).velocity) < -Para.N_c/(Para.Nt.tot*2)
    %     TargetList.velocity = TargetList.velocity + Para.N_c/Para.Nt.tot;
    % end
end
% TargetList.velocity = velocity_cum/count;   
TargetList.range_average = mean(TargetList.range);
TargetList.velocity_average = mean(TargetList.velocity);

velocity_average = est_velocity - IM_vel_est + IM_vel_flag*Para.N_c/Para.Nt.tot;
%% Determine Range, Velocity and Angle
if mod(i_frame,Para.N_period) ~= 9   
    % velocity_all = mod(round(TargetList.velocity_average)-1 + (0:Para.N_c/Para.Nt.tot:Para.N_c-Para.N_c/Para.Nt.tot), ...
    %                Para.N_c)+1;  
    velocity_all = mod(round(velocity_average)-1 + (0:Para.N_c/Para.Nt.tot:Para.N_c-Para.N_c/Para.Nt.tot), ...
                   Para.N_c)+1;

    % Determine angle
    X2_Detected = circshift(X2_Detected, -(IM_rng_est));
    X = circshift(X, -(IM_rng_est));
    if i_frame > Para.N_period*Para.N_period_VelStartTxInfo
        X2_Detected = circshift(X2_Detected, -(IM_vel_est), 2);
        X = circshift(X, -(IM_vel_est), 2);
    end

    TargetList = DetermineRangeVelocityAngle( ...
            TargetList, ...
            X, ...
            X2_Detected, ...
            Para.Nt, ...
            Para.Nr, ...
            Para.N_c, ...
            Para.FoV, ...
            Para.AntSpa, ...
            'comm');
    TargetList.range_average = mean(TargetList.range);    
    TargetList.velocity_average = mean(TargetList.velocity);
    TargetList.azi_average = mean(TargetList.azi);
    % azi_tx_average = mean(TargetList.azi_tx);
    
    % Compensate Delay and Velocity
    exp_factor = exp(-1j * 2 * pi * (0:Para.N_f-1) * predict_relative_r / Para.N_f).';
    TDData_B(:,1:Para.N_c,1:Para.Nr.tot) = TDData_B(:,1:Para.N_c,1:Para.Nr.tot) .* exp_factor;
    X_nowin = zeros(Para.N_f, Para.N_c, Para.Nr.tot);
    for i_r = 0:Para.Nr.tot-1
        X_nowin(:,:,i_r+1) = fftshift(fft(fft(TDData_B(:,:,i_r+1).*Para.W_f).*Para.W_c...
                       , Para.N_c, 2),...
                       2);
    end    
    % Determine QAM
    Gain_RDM_cur = zeros(Para.Nt.tot, Para.Nr.tot);
    for i = 1:Para.Nt.tot
        for j = 1:Para.Nr.tot
            Gain_RDM_cur(i,j) = X_nowin(round(range_average)+IM_rng_est, ...
                            mod(velocity_all(i)+IM_vel_est-1, Para.N_c)+1, ...
                            j)...
            *...
                        exp(-1j*2*pi*(Para.Nt.d_azi/Para.lambda*sind(predict_azi_tx)*cosd(0)*mod(i-1, Para.Nt.azi)))*...% Tx azimuth
                        exp(-1j*2*pi*(Para.Nt.d_ele/Para.lambda*sind(0)*floor((i-1)/Para.Nt.azi)))*...% Tx elevation
                        exp(-1j*2*pi*(Para.Nr.d_azi/Para.lambda*sind(predict_azi)*cosd(0)*mod(j-1, Para.Nr.azi)))*...% Rx azimuth
                        exp(-1j*2*pi*(Para.Nr.d_ele/Para.lambda*sind(0)*floor((j-1)/Para.Nr.azi)));% Rx elevation;
        end
    end
%     lzr = Gain_RDM_cur./Gain_RDM_pre;
%     figure;plot(lzr(:), 'ro');
%     xlim([-2, 2]), ylim([-2, 2])
    if i_frame <= Para.N_period*Para.N_period_QAMStartTxInfo  || mod(i_frame, 10)==0
        QAM_est = 1;
    else
        QAM_est = DetermineQAM(mean(Gain_RDM_cur./Gain_RDM_pre, 'all')); 
        if mod(range_average_pre,1) < 0.5 && mod(range_average,1) >=0.5 && range_average >  range_average_pre ...
                || ...
                mod(range_average_pre,1) >= 0.5 && mod(range_average,1) <0.5 && range_average <  range_average_pre
            QAM_est = -1*QAM_est;
        end
        if mod(velocity_average_pre(2),1) < 0.5 && mod(velocity_average,1) >=0.5 &&  velocity_average > velocity_average_pre(2)   ...
                ||...
               mod(velocity_average_pre(2),1) >= 0.5 && mod(velocity_average,1) <0.5 &&  velocity_average < velocity_average_pre(2)
            QAM_est = -1*QAM_est;
        end
    end
    
    % if 
    Gain_RDM_cur = Gain_RDM_cur/QAM_est; 

    % Determine Label
    TargetList.label = 'A_0';

    % Determine Orientation          
    if i_frame == 1
        TargetList.orientation = TargetList_pre(index).orientation;
        TargetList.orientationVar = TargetList_pre(index).orientationVar;
    else
        TargetList = DetermineOrientation_comm( ...
                            TargetList_pre_All, ...
                            TargetList);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
else% mod(i,Para.N_period) == 9
    TargetList.range_average = mean(TargetList.range);    
    TargetList.velocity_average = mean(TargetList.velocity);

%     X2_Detected = circshift(X2_Detected, -DstOffset_est);

    % Determine QAM 
    Gain_RDM_cur = Gain_RDM_pre;
    QAM_est = 1;
    

    % Determine angle
    % A's angle can't be determined and we resort to estimate it by using
    % previous results
    TargetList_pre2 = TargetList_pre_All(end-1).TargetList;
    
    % obtain all the label of the previous TargetList
    AllLabel_cell = {TargetList_pre2(:).label};
    AllLabel_array = squeeze(char(( AllLabel_cell )));
    tmp = repmat('A_0', size(AllLabel_array,1), 1) ...
        == AllLabel_array;
    index2 = all(tmp, 2);    

    TargetList.azi = 2*TargetList_pre(index).azi ...
        - TargetList_pre2(index2).azi_average;
    TargetList.azi_average = 2*TargetList_pre(index).azi_average ...
        - TargetList_pre2(index2).azi_average;
    TargetList.ele = 2*TargetList_pre(index).ele ...
        - mean(TargetList_pre2(index2).ele);
    TargetList.ele_average = 2*TargetList_pre(index).ele_average ...
        - mean(TargetList_pre2(index2).ele);

    % Determine orientation
    TargetList.label = 'A_0';
%     TargetList.orientation = TargetList_pre(index).orientation;
%     TargetList.orientationVar = TargetList_pre(index).orientationVar; 
    TargetList = DetermineOrientation_comm( ...
                        TargetList_pre_All, ...
                        TargetList);
    
end
% DstOffset_est = DstOffset_est + 1;
% figure;mesh(angle(squeeze(Gain_RDM_cur))/pi*80)
% figure;mesh(abs(squeeze(Gain_RDM_cur)))
end

function QAM_est = DetermineQAM(data)

    codebook_angle = [0 180 -180];
    codebook_QAM = [1 -1 -1];
    result = abs(angle(data)/pi*180 + codebook_angle);
    [~, index] = min(result);
    QAM_est = codebook_QAM(index);

    % codebook_angle = [45 135 -135 -45];
    % codebook_QAM = sqrt(2)/2*[1+1j,  1-1j,  1+1j,  1-1j];
    % result = abs(angle(data)/pi*180 - codebook_angle);
    % [~, index] = min(result);
    % QAM_est = codebook_QAM(index);
end