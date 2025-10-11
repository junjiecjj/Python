%% Preliminary
clear; clc
% close all
rng(666, 'twister')
addpath('Function\')

if isempty(gcp('nocreate'))   
    numWorkers = 10;      
    parpool('local', numWorkers);
end

Para = ParaClass_320_512();
SNRdB = -25:-1:-35;
% SNRdB = -20;
Nit = 1e4;
NumQAMBit = 2;
NumIMRngBit = log2(Para.TcEff*Para.fs)-1;
NumIMVelBit = log2(Para.N_c/Para.Nt.tot);
QAM_est = zeros(length(SNRdB), Nit);
IM_rng_est = zeros(length(SNRdB), Nit);
IM_vel_est = zeros(length(SNRdB), Nit);
rng_error = zeros(length(SNRdB), Nit);
vel_error = zeros(length(SNRdB), Nit);
azi_error = zeros(length(SNRdB), Nit);
ele_error = zeros(length(SNRdB), Nit);

[QAMsymbols, QAMbits] = generateQAM(2^NumQAMBit);
q_deAmb = LinearPhase_Gen_DDM( ...
            Para.Nt.tot,...
            Para.N_c,...
            1);% Generate the DDMA sequence, Nt × Nc, divide N_c into Nt+1 parts
q = LinearPhase_Gen_DDM( ...
        Para.Nt.tot, ...
        Para.N_c, ...
        2);% Nt × Nc, divide N_c into Nt parts

HitRate(1:length(SNRdB)) = struct(...
    'RangeVelocity', 0,...
    'RangeVelocityAngle',0);
BER(1:length(SNRdB)) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
SER(1:length(SNRdB)) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
%% Bit Generation
DataAll = DataGen_BitErrorRate(Nit, NumQAMBit, NumIMRngBit, NumIMVelBit);
for i_SNR = 1:length(SNRdB)
    for i_it = 1:Nit
        tic
        Para = Para.ParaSet();
        %% Time-domain data generation
        % Calculate the energy of TDData in advance to reduce computation
        TDData = zeros(Para.N_f, ...
            Para.N_c, ...
            Para.Nr.tot);% received data 
        for i_r = 0:Para.Nr.tot-1
            for i_c = 0:Para.N_c-1
                for i_t = 0:Para.Nt.tot-1                                   % DDMA, first azi, then ele
                    TDData(:,i_c+1, i_r+1) = TDData(:,i_c+1, i_r+1) + ...
                    Para.Target_A(1).a *...                      % Amplitude
                    exp(1j*2*pi*((0:Para.N_f-1)* ...
                    (Para.Target_A(1).fr)...
                    /Para.N_f)).'*...                                       % Fast time
                    exp(1j*2*pi*(i_c*Para.Target_A(1).fv/Para.N_c))*...     % Slow time
                    q(i_t+1, i_c+1)*...                                     % DDMA
                    exp(1j*2*pi*(Para.Nt.d_azi/Para.lambda*sind(Para.Target_A(1).azi)*cosd(Para.Target_A(1).ele)*mod(i_t, Para.Nt.azi)))*...% Tx azimuth
                    exp(1j*2*pi*(Para.Nt.d_ele/Para.lambda*sind(Para.Target_A(1).ele)*floor(i_t/Para.Nt.azi)))*...                          % Tx elevation
                    exp(1j*2*pi*(Para.Nr.d_azi/Para.lambda*sind(Para.Target_A(1).azi)*cosd(Para.Target_A(1).ele)*mod(i_r, Para.Nr.azi)))*...% Rx azimuth
                    exp(1j*2*pi*(Para.Nr.d_ele/Para.lambda*sind(Para.Target_A(1).ele)*floor(i_r/Para.Nr.azi)));                             % Rx elevation
                end           
            end
        end
        % Generate Gain_RDM_pre to obtain QAM
        Gain_RDM_pre = zeros(Para.Nt.tot, Para.Nr.tot);
        RealRange = Para.Target_A(1).fr + 1;
        RealVelocity = Para.Target_A(1).fv + 1;
        velocity_all = mod(round(RealVelocity-1 ...
            + (0:Para.N_c/Para.Nt.tot:Para.N_c-Para.N_c/Para.Nt.tot)),...
            Para.N_c) + 1;
        X = zeros(Para.N_f, Para.N_c, Para.Nr.tot);
        for i_r = 0:Para.Nr.tot-1
            % if there is more than 1 Rx antenna, X2 need non-coherent
            X(:,:,i_r+1) = fftshift(fft( ...
                    fft(TDData(:,:,i_r+1).*Para.W_f).*Para.W_c, ...
                    Para.N_c, ...
                    2),...
                    2);    
        end
        for i = 1:Para.Nt.tot
            for j = 1:Para.Nr.tot
                Gain_RDM_pre(i,j) = X(round(RealRange), ...
                                velocity_all(i), ...
                                j);
            end
        end
        TDData = zeros(Para.N_f, ...
                        Para.N_c, ...
                        Para.Nr.tot);% received data           
        for i_r = 0:Para.Nr.tot-1
            for i_c = 0:Para.N_c-1
                for i_t = 0:Para.Nt.tot-1                                   % DDMA, first azi, then ele
                    TDData(:,i_c+1, i_r+1) = TDData(:,i_c+1, i_r+1) + ...
                    Para.Target_A(1).a * DataAll.QAM(i_it)*...                      % Amplitude
                    exp(1j*2*pi*((0:Para.N_f-1)* ...
                    (Para.Target_A(1).fr + DataAll.IM_rng(i_it))...
                    /Para.N_f)).'*...                                       % Fast time
                    exp(1j*2*pi*(i_c*(Para.Target_A(1).fv + DataAll.IM_vel(i_it))/Para.N_c))*...     % Slow time
                    q_deAmb(i_t+1, i_c+1)*...                                     % DDMA
                    exp(1j*2*pi*(Para.Nt.d_azi/Para.lambda*sind(Para.Target_A(1).azi)*cosd(Para.Target_A(1).ele)*mod(i_t, Para.Nt.azi)))*...% Tx azimuth
                    exp(1j*2*pi*(Para.Nt.d_ele/Para.lambda*sind(Para.Target_A(1).ele)*floor(i_t/Para.Nt.azi)))*...                          % Tx elevation
                    exp(1j*2*pi*(Para.Nr.d_azi/Para.lambda*sind(Para.Target_A(1).azi)*cosd(Para.Target_A(1).ele)*mod(i_r, Para.Nr.azi)))*...% Rx azimuth
                    exp(1j*2*pi*(Para.Nr.d_ele/Para.lambda*sind(Para.Target_A(1).ele)*floor(i_r/Para.Nr.azi)));                             % Rx elevation
                end           
            end
        end
        TDData = awgn(TDData, SNRdB(i_SNR),'measured');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        TDData_angle = zeros(Para.N_f, ...
                        Para.N_c, ...
                        Para.Nr.tot);% received data           
        for i_r = 0:Para.Nr.tot-1
            for i_c = 0:Para.N_c-1
                for i_t = 0:Para.Nt.tot-1                                   % DDMA, first azi, then ele
                    TDData_angle(:,i_c+1, i_r+1) = TDData_angle(:,i_c+1, i_r+1) + ...
                    Para.Target_A(1).a * DataAll.QAM(i_it)*...                      % Amplitude
                    exp(1j*2*pi*((0:Para.N_f-1)* ...
                    (Para.Target_A(1).fr + DataAll.IM_rng(i_it))...
                    /Para.N_f)).'*...                                       % Fast time
                    exp(1j*2*pi*(i_c*(Para.Target_A(1).fv + DataAll.IM_vel(i_it))/Para.N_c))*...     % Slow time
                    q(i_t+1, i_c+1)*...                                     % DDMA
                    exp(1j*2*pi*(Para.Nt.d_azi/Para.lambda*sind(Para.Target_A(1).azi)*cosd(Para.Target_A(1).ele)*mod(i_t, Para.Nt.azi)))*...% Tx azimuth
                    exp(1j*2*pi*(Para.Nt.d_ele/Para.lambda*sind(Para.Target_A(1).ele)*floor(i_t/Para.Nt.azi)))*...                          % Tx elevation
                    exp(1j*2*pi*(Para.Nr.d_azi/Para.lambda*sind(Para.Target_A(1).azi)*cosd(Para.Target_A(1).ele)*mod(i_r, Para.Nr.azi)))*...% Rx azimuth
                    exp(1j*2*pi*(Para.Nr.d_ele/Para.lambda*sind(Para.Target_A(1).ele)*floor(i_r/Para.Nr.azi)));                             % Rx elevation
                end           
            end
        end        
        TDData_angle = awgn(TDData_angle, SNRdB(i_SNR),'measured');
%% Freq-domain data generation
        X2_sum = zeros(Para.N_f, Para.N_c);
        for i_r = 0:Para.Nr.tot-1
            % if there is more than 1 Rx antenna, X2 need non-coherent
            X = fftshift(fft( ...
                    fft(TDData(:,:,i_r+1).*Para.W_f).*Para.W_c, ...
                    Para.N_c, ...
                    2),...
                    2);
            X2_sum = X2_sum + X.*conj(X);    
        end
        X2_sum = X2_sum/Para.Nr.tot;
        X2_Detected = CFAR(2, X2_sum, Para.RDM_CFAR, Para.SNRdB, 'CA');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X2_sum_angle = zeros(Para.N_f, Para.N_c);
        X = zeros(Para.N_f, Para.N_c, Para.Nr.tot);
        for i_r = 0:Para.Nr.tot-1
            % if there is more than 1 Rx antenna, X2 need non-coherent
            X(:,:,i_r+1) = fftshift(fft( ...
                    fft(TDData_angle(:,:,i_r+1).*Para.W_f).*Para.W_c, ...
                    Para.N_c, ...
                    2),...
                    2);
            X2_sum_angle = X2_sum_angle + X(:,:,i_r+1).*conj(X(:,:,i_r+1));    
        end
        X2_sum_angle = X2_sum_angle/Para.Nr.tot;
        X2_Detected_angle = CFAR(2, X2_sum_angle, Para.RDM_CFAR, Para.SNRdB, 'CA');
%% Detection
        TargetList = DetermineRangeVelocity(X2_Detected, Para.N_c, Para.Nt.tot);
        if numel(TargetList) > 3
            error('False alarm propobility is too higy!!!')
        end
        TargetList_angle = DetermineRangeVelocityAngle( ...
                    TargetList, ...
                    X, ...
                    X2_Detected_angle, ...
                    Para.Nt, ...
                    Para.Nr, ...
                    Para.N_c, ...
                    Para.FoV, ...
                    Para.AntSpa, ...
                    'radar');
%% Estimation         
        if isempty([TargetList_angle.range])
            QAM_est(i_SNR, i_it) = -1;
            IM_rng_est(i_SNR, i_it) = -1;
            IM_vel_est(i_SNR, i_it) = -1;

            SER(i_SNR).NumQAMError = SER(i_SNR).NumQAMError + 1;
            SER(i_SNR).NumIMRngError = SER(i_SNR).NumIMRngError + 1;
            SER(i_SNR).NumIMVelError = SER(i_SNR).NumIMVelError + 1;

            BER(i_SNR).NumQAMError = BER(i_SNR).NumQAMError + NumQAMBit;
            BER(i_SNR).NumIMRngError = BER(i_SNR).NumIMRngError + NumIMRngBit;
            BER(i_SNR).NumIMVelError = BER(i_SNR).NumIMVelError + NumIMVelBit;
        else
            IM_rng_est(i_SNR, i_it) = round(mean(TargetList_angle(1).range) - RealRange);
            IM_vel_est(i_SNR, i_it) = round(mean(TargetList_angle(1).velocity) - (Para.N_c/2) - RealVelocity);
            rng_error(i_SNR, i_it) = mean(TargetList_angle(1).range) - DataAll.IM_rng(i_it) - RealRange;
            vel_error(i_SNR, i_it) = mean(TargetList_angle(1).velocity) - (Para.N_c/2) - DataAll.IM_vel(i_it) - RealVelocity;
            azi_error(i_SNR, i_it) = mean(TargetList_angle(1).azi) - Para.Target_A(1).azi;
            ele_error(i_SNR, i_it) = mean(TargetList_angle(1).ele) - Para.Target_A(1).ele;

            Gain_RDM_cur = zeros(Para.Nt.tot, Para.Nr.tot);
            velocity_all = mod(round(RealVelocity+IM_vel_est(i_SNR, i_it)-1 ...
                + (0:Para.N_c/Para.Nt.tot:Para.N_c-Para.N_c/Para.Nt.tot)),...
                Para.N_c) + 1;
            for i = 1:Para.Nt.tot
                for j = 1:Para.Nr.tot
                    Gain_RDM_cur(i,j) = X(round(RealRange)+IM_rng_est(i_SNR, i_it), ...
                                    velocity_all(i), ...
                                    j);
                end
            end
            [QAM_est(i_SNR, i_it), QAMbits_est] = DetermineQAM(mean(Gain_RDM_cur./Gain_RDM_pre, 'all'), QAMsymbols, QAMbits);
            % Prevent exceeding the range
            if IM_rng_est(i_SNR, i_it) < 0
                IM_rng_est(i_SNR, i_it) = 0;
            elseif IM_rng_est(i_SNR, i_it) >= 2^NumIMRngBit
                IM_rng_est(i_SNR, i_it) = 2^NumIMRngBit-1;
            end
            if IM_vel_est(i_SNR, i_it) < 0
                IM_vel_est(i_SNR, i_it) = 0;
            elseif IM_vel_est(i_SNR, i_it) >= 2^NumIMVelBit
                IM_vel_est(i_SNR, i_it) = 2^NumIMVelBit-1;
            end
            % Convert QAM and IM to QAM_bit and IM_bit
            [IM_Rng_bit_est, IM_Vel_bit_est] = Convert2Bit(IM_rng_est(i_SNR, i_it), NumIMRngBit, IM_vel_est(i_SNR, i_it), NumIMVelBit);
%% Save and Print
            % Hitrate
            if abs(mean(TargetList_angle(1).range) - IM_rng_est(i_SNR, i_it) - RealRange) <= 1 ...
                && abs(mean(TargetList_angle(1).velocity)-Para.N_c/2 - DataAll.IM_vel(i_it) - RealVelocity) <= 1
                
                HitRate(i_SNR).RangeVelocity = HitRate(i_SNR).RangeVelocity + 1;
                if abs(mean(TargetList_angle(1).azi) - Para.Target_A(1).azi) <= 3                
                    HitRate(i_SNR).RangeVelocityAngle = HitRate(i_SNR).RangeVelocityAngle + 1;
                end
            end  
            
            % SER and BER
            if abs(QAM_est(i_SNR, i_it) - DataAll.QAM(i_it)) > sqrt(2)/2
                SER(i_SNR).NumQAMError = SER(i_SNR).NumQAMError + 1;
                BER(i_SNR).NumQAMError = BER(i_SNR).NumQAMError + sum(xor(QAMbits_est, DataAll.QAM_bit(((i_it-1)*NumQAMBit + 1) : i_it*NumQAMBit)));
            end
            if IM_rng_est(i_SNR, i_it) - DataAll.IM_rng(i_it) ~= 0
                SER(i_SNR).NumIMRngError = SER(i_SNR).NumIMRngError + 1;
                BER(i_SNR).NumIMRngError = BER(i_SNR).NumIMRngError + sum( xor(IM_Rng_bit_est, ...
                    DataAll.IM_rng_bit(((i_it-1)*NumIMRngBit + 1) : i_it*NumIMRngBit) ) ...
                    );
            end
            if IM_vel_est(i_SNR, i_it) - DataAll.IM_vel(i_it) ~= 0
                SER(i_SNR).NumIMVelError = SER(i_SNR).NumIMVelError + 1;
                BER(i_SNR).NumIMVelError = BER(i_SNR).NumIMVelError + sum( xor(IM_Vel_bit_est, ...
                    DataAll.IM_vel_bit(((i_it-1)*NumIMVelBit + 1) : i_it*NumIMVelBit) ) ...
                    );
            end
        end

        % Print
        fprintf('---------------------------------------------------------\n');
        fprintf('SNRdB  = %4ddB,\n', SNRdB(i_SNR));
        fprintf('it/Nit = %8d/%8d = %5.2f%%\n', i_it, Nit, i_it/Nit*100);
        fprintf('\n');
        fprintf('QAM            SER         %8d/%8d = %8.7f\n', SER(i_SNR).NumQAMError, i_it, SER(i_SNR).NumQAMError/i_it);
        fprintf('IM  range      SER         %8d/%8d = %8.7f\n', SER(i_SNR).NumIMRngError, i_it, SER(i_SNR).NumIMRngError/i_it);        
        fprintf('IM  velocity   SER         %8d/%8d = %8.7f\n', SER(i_SNR).NumIMVelError, i_it, SER(i_SNR).NumIMVelError/i_it);
        fprintf('\n');
        fprintf('QAM            BER         %8d/%8d = %8.7f\n', BER(i_SNR).NumQAMError, (i_it * NumQAMBit), BER(i_SNR).NumQAMError/(i_it * NumQAMBit));                
        fprintf('IM  range      BER         %8d/%8d = %8.7f\n', BER(i_SNR).NumIMRngError, (i_it * NumIMRngBit), BER(i_SNR).NumIMRngError/(i_it * NumIMRngBit));
        fprintf('IM  velocity   BER         %8d/%8d = %8.7f\n', BER(i_SNR).NumIMVelError, (i_it * NumIMVelBit), BER(i_SNR).NumIMVelError/(i_it * NumIMVelBit));        
        fprintf('\n');
        fprintf('Hitrate RangeVelocity      %8d/%8d = %8.7f\n', HitRate(i_SNR).RangeVelocity, i_it, HitRate(i_SNR).RangeVelocity/i_it)
        fprintf('Hitrate RangeVelocityAngle %8d/%8d = %8.7f\n', HitRate(i_SNR).RangeVelocityAngle, i_it, HitRate(i_SNR).RangeVelocityAngle/i_it)
        fprintf('\n');
        fprintf('range error(unit reso)     %8.7f\n', rng_error(i_SNR, i_it));
        fprintf('velocity error(unit reso)  %8.7f\n', vel_error(i_SNR, i_it));
        fprintf('azi error(deg)             %8.7f\n', azi_error(i_SNR, i_it));
        toc
    end
    save('.\Data\B_320_T_512.mat', ...
        'SNRdB', ...
        'Nit', ...
        'NumQAMBit', ...
        'NumIMRngBit', ...
        'NumIMVelBit', ...
        'HitRate', ...
        'BER', ...
        'SER', ...
        'QAM_est', ...
        'IM_rng_est', ...
        'IM_vel_est', ...
        'rng_error',...
        'vel_error',...
        'azi_error',...
        'ele_error',...
        'DataAll')
end

function [QAM_est, QAMbits_est] = DetermineQAM(data, QAMsymbols, QAMbits)
    [~, idx] = min(abs(data - QAMsymbols));
    QAM_est = QAMsymbols(idx);
    QAMbits_est = QAMbits(idx,:);
end
function [IM_rng_bit_est, IM_vel_bit_est] = Convert2Bit(IM_rng_est, NumIMRngBit, IM_vel_est, NumIMVelBit)
% input
%       QAM_est         1
%       IM_est          1
% output
%       IM_bit_est      NumIMBit×1
     IM_rng_bit_est = dec2bin(IM_rng_est, NumIMRngBit);
     IM_vel_bit_est = dec2bin(IM_vel_est, NumIMVelBit);
end