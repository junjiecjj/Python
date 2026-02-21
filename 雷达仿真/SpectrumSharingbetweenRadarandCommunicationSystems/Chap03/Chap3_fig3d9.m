% Figure 3.9 - NSP SSVSP best worst versus number of radar antennas (M_R)
clear all;
close all;
clc;

% Parameters
C = 4; % No of clusters
K = 3; % No of BS in the cluster
M_k = [6 5 4 3]; % Different antenna configurations for different clusters
N_k = [6 5 4 3]; % Different antenna configurations for different clusters
M_R = 20:30; % No of transmit antenna elements on RADAR
N_R = 20:30; % No of receive antenna elements on RADAR
L_b = 512; % block length
L_t = 256;
theta = 0; % target direction in degrees
fc = 3.5e9; % frequency of operation in Hz
c = 3e8; % speed of light in m/s
lambda = c / fc;
d = 3 * lambda / 4; % inter-element spacing on radar
SNR = 20; % SNR of received radar signal from target in dB
snr = 10^(SNR/10);
DoF = 1:max(M_k);
rho_db = 15;

% Preallocate arrays for storing results
crb = zeros(1, length(M_R));
crb_null_best = zeros(1, length(M_R));
crb_null_worst = zeros(1, length(M_R));
crb_ssvsp_best = zeros(1, length(M_R));
crb_ssvsp_worst = zeros(1, length(M_R));

% Main simulation loop
for n = 1:length(M_R)
    % Steering vectors and their differentiations
    a_t = steering_ULA(theta, M_R(n), fc, d);
    a_r = steering_ULA(theta, N_R(n), fc, d);
    a_tdiff = diff_steering_ULA(theta, M_R(n), fc, d);
    a_rdiff = diff_steering_ULA(theta, N_R(n), fc, d);
    
    % Generate radar signals
    s = (1/sqrt(2)) * (hadamard(L_b) + 1i * hadamard(L_b));
    s1 = s(1:M_R(n), :);
    Rs = (1/L_b) * s1 * s1';
    % Preallocate arrays for Monte Carlo simulation
    cb_temp = zeros(1, 100);
    cb_best_nsp_temp = zeros(1, 100);
    cb_worst_nsp_temp = zeros(1, 100);
    cb_best_vsp_temp = zeros(1, 100);
    cb_worst_vsp_temp = zeros(1, 100);
    for x = 1:100
        % Initialize memory for all clusters
        H_mem = zeros(K * max(M_k), C * M_R(n));
        % Initialize arrays for metrics
        nltv = zeros(1, C); % Null space dimension for each cluster
        metric = zeros(1, C); % SSVSP metric for each cluster
        for p = 1:C
            d_k = M_k(p) * ones(1, K); % DoF for each BS in this cluster
            d_ksum = sum(d_k);
            % Composite channel generation for this cluster
            H_bar = zeros(M_R(n), d_ksum);
            for a = 1:K
                H_kR = randn(d_k(a), M_R(n)) + 1i * randn(d_k(a), M_R(n));
                if (a == 1)
                    dim = 1:d_k(a);
                else
                    dim = (sum(d_k(1:a-1))+1):(sum(d_k(1:a)));
                end
                H_bar(:, dim) = H_kR';
            end
            H_real = H_bar';
            % Store in memory
            H_mem(1:K*M_k(p), (p-1)*M_R(n)+1:p*M_R(n)) = H_real;
            % Null-space computation for perfect CSI (NSP)
            [U_per, B_per, V_per] = svd(H_real); % SVD of H
            % Select eigenvectors corresponding to SVs below threshold sigma
            if d_ksum == 1
                B1 = B_per(d_ksum, d_ksum);
            else
                B1 = diag(B_per)';
            end
            [row1, col1] = find(B1 == 0); % Find cols with SV < threshold
            col = [col1 d_ksum+1:M_R(n)];
            nltv(p) = length(col); % Store null space dimension
            % SSVSP computation
            ss1 = length(B1);
            colms = [ss1 d_ksum+1:M_R(n)];
            V_tildalms = V_per(:, colms); % Pick corresponding cols of V
            P_rlms = V_tildalms * V_tildalms';
            % Compute SSVSP metric
            sig_rad = s(1:M_R(n), 1);
            diff = P_rlms * sig_rad - sig_rad;
            metric(p) = norm(diff);
        end
        
        % Find best and worst cases for NSP based on null space dimension
        [best_nsp, I_best_nsp] = max(nltv); % Largest null space
        [worst_nsp, I_worst_nsp] = min(nltv); % Smallest null space
        
        % Find best and worst cases for SSVSP based on metric
        [best_vsp_metric, I_best_vsp] = min(metric); % Smallest error
        [worst_vsp_metric, I_worst_vsp] = max(metric); % Largest error
        
        % Extract best and worst channel matrices for NSP
        H_best_nsp = H_mem(1:K*M_k(I_best_nsp), (I_best_nsp-1)*M_R(n)+1:I_best_nsp*M_R(n));
        H_worst_nsp = H_mem(1:K*M_k(I_worst_nsp), (I_worst_nsp-1)*M_R(n)+1:I_worst_nsp*M_R(n));
        
        % Extract best and worst channel matrices for SSVSP
        H_best_vsp = H_mem(1:K*M_k(I_best_vsp), (I_best_vsp-1)*M_R(n)+1:I_best_vsp*M_R(n));
        H_worst_vsp = H_mem(1:K*M_k(I_worst_vsp), (I_worst_vsp-1)*M_R(n)+1:I_worst_vsp*M_R(n));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  NSP - Best case
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        d_k_best_nsp = M_k(I_best_nsp) * ones(1, K); % DoF
        d_ksum_best_nsp = sum(d_k_best_nsp);
        
        [U_bn, B_bn, V_bn] = svd(H_best_nsp); % SVD of H
        
        if d_ksum_best_nsp == 1
            B1_bn = B_bn(d_ksum_best_nsp, d_ksum_best_nsp);
        else
            B1_bn = diag(B_bn)';
        end
        
        [rowbn, colbn] = find(B1_bn == 0); % Find cols with SV < threshold
        cobn = [colbn d_ksum_best_nsp+1:M_R(n)];
        V_tildabn = V_bn(:, cobn); % Pick corresponding cols of V
        P_Rbn = V_tildabn * V_tildabn';
        Rs_bn = P_Rbn * P_Rbn';
        
        cb_temp(x) = CRB(Rs, a_t, a_tdiff, a_rdiff, M_R(n), snr); % orthogonal signals
        cb_best_nsp_temp(x) = CRB(Rs_bn, a_t, a_tdiff, a_rdiff, M_R(n), snr);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  NSP - Worst case
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        d_k_worst_nsp = M_k(I_worst_nsp) * ones(1, K); % DoF
        d_ksum_worst_nsp = sum(d_k_worst_nsp);
        
        [U_wn, B_wn, V_wn] = svd(H_worst_nsp); % SVD of H
        
        if d_ksum_worst_nsp == 1
            B1_wn = B_wn(d_ksum_worst_nsp, d_ksum_worst_nsp);
        else
            B1_wn = diag(B_wn)';
        end
        
        [rowwn, colwn] = find(B1_wn == 0); % Find cols with SV < threshold
        cown = [colwn d_ksum_worst_nsp+1:M_R(n)];
        V_tildawn = V_wn(:, cown); % Pick corresponding cols of V
        P_Rwn = V_tildawn * V_tildawn';
        Rs_wn = P_Rwn * P_Rwn';
        
        cb_worst_nsp_temp(x) = CRB(Rs_wn, a_t, a_tdiff, a_rdiff, M_R(n), snr);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  SSVSP - Best case
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        d_k_best_vsp = M_k(I_best_vsp) * ones(1, K); % DoF
        d_ksum_best_vsp = sum(d_k_best_vsp);
        
        [U_bv, B_bv, V_bv] = svd(H_best_vsp); % SVD of H
        
        if d_ksum_best_vsp == 1
            B1_bv = B_bv(d_ksum_best_vsp, d_ksum_best_vsp);
        else
            B1_bv = diag(B_bv)';
        end
        
        ss1_bv = length(B1_bv);
        covb = [ss1_bv d_ksum_best_vsp+1:M_R(n)];
        V_tildabv = V_bv(:, covb); % Pick corresponding cols of V
        P_Rbv = V_tildabv * V_tildabv';
        Rs_bv = P_Rbv * P_Rbv';
        cb_best_vsp_temp(x) = CRB(Rs_bv, a_t, a_tdiff, a_rdiff, M_R(n), snr);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  SSVSP - Worst case
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        d_k_worst_vsp = M_k(I_worst_vsp) * ones(1, K); % DoF
        d_ksum_worst_vsp = sum(d_k_worst_vsp);
        
        [U_wv, B_wv, V_wv] = svd(H_worst_vsp); % SVD of H
        if d_ksum_worst_vsp == 1
            B1_wv = B_wv(d_ksum_worst_vsp, d_ksum_worst_vsp);
        else
            B1_wv = diag(B_wv)';
        end
        ss1_wv = length(B1_wv);
        covw = [ss1_wv d_ksum_worst_vsp+1:M_R(n)];
        V_tildawv = V_wv(:, covw); % Pick corresponding cols of V
        P_Rvw = V_tildawv * V_tildawv';
        Rs_wv = P_Rvw * P_Rvw';
        cb_worst_vsp_temp(x) = CRB(Rs_wv, a_t, a_tdiff, a_rdiff, M_R(n), snr);
    end
    % Calculate average values for this M_R value
    crb(n) = mean(cb_temp);
    crb_null_best(n) = mean(cb_best_nsp_temp);
    crb_null_worst(n) = mean(cb_worst_nsp_temp);
    crb_ssvsp_best(n) = mean(cb_best_vsp_temp);
    crb_ssvsp_worst(n) = mean(cb_worst_vsp_temp);
end

% Plot results
figure(1)
semilogy(M_R, crb, '-kh', 'LineWidth', 2)
hold on
semilogy(M_R, abs(crb_null_best), '-rs', 'LineWidth', 2)
hold on
semilogy(M_R, abs(crb_null_worst), '-go', 'LineWidth', 2)
hold on
semilogy(M_R, abs(crb_ssvsp_best), '-bx', 'LineWidth', 2)
hold on
semilogy(M_R, abs(crb_ssvsp_worst), '-cd', 'LineWidth', 2)
xlabel('Number of radar antennas (M_R)', 'fontsize', 14)
ylabel('RMSE (degree)', 'fontsize', 14)
legend('orthogonal signals', 'NSP best', 'NSP worst', 'SSVSP best', 'SSVSP worst', 'Location', 'Best')
grid on;
title('NSP and SSVSP Performance vs Number of Radar Antennas');
hold off;