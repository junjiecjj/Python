% Figure 3.8 - Radar Interference versus number of BSs/cluster
clear all; close all; clc;

% Parameters
K = 1:10; % No of BSs per cluster
M_k = 6 * ones(1, length(K));
N_k = 6 * ones(1, length(K));
M_R = 100; % No of transmit antenna elements on RADAR
N_R = 100; % No of receive antenna elements on RADAR
L_b = 512; % block length
L_t = [64 256]; % part of block used for channel estimation, has to be power of 2 for using 'hadamard'
SNR = 20; % SNR of received radar signal from target in dB
snr = 10^(SNR/10);
rho_db = 20;
rho_ratio = 10^(rho_db/10);
noise_v = sqrt(1/rho_ratio);

% Preallocate arrays for storing results
frob_null_per = zeros(1, length(K));
frob_null_wchest1 = zeros(1, length(K));
frob_null_wchest2 = zeros(1, length(K));
frob_ssvsp_per = zeros(1, length(K));
frob_ssvsp_wchest1 = zeros(1, length(K));
frob_ssvsp_wchest2 = zeros(1, length(K));

% ESTIMATION OF COMPOSITE CHANNEL FROM COMMUNICATION RECEIVER TO RADAR TRANSMITTER
for n = 1:length(K)
    d_k = 6 * ones(1, n); % DoF for kth user
    d_ksum = sum(d_k);
    
    % Preallocate temporary arrays for Monte Carlo simulation
    frob_per_temp = zeros(1, 100);
    frob_perms_temp = zeros(1, 100);
    frob_wchest_temp = zeros(2, 100); % 2 rows for L_t(1) and L_t(2)
    frob_wchestms_temp = zeros(2, 100);
    
    for x = 1:100
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Composite channel generation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        H_kR = zeros(N_k(1), n * M_R);
        H_bar = zeros(M_R, d_ksum);
        
        for a = 1:n
            H_kR(:, (a-1)*M_R+1:a*M_R) = randn(N_k(a), M_R) + 1i * randn(N_k(a), M_R);
            if (a == 1)
                dim = 1:d_k(a);
            else
                dim = (sum(d_k(1:a-1))+1):(sum(d_k(1:a)));
            end
            H_bar(:, dim) = (H_kR(:, (a-1)*M_R+1:a*M_R))';
        end
        H_real = H_bar';
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Null-space computation for perfect CSI
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [U_per, B_per, V_per] = svd(H_real); % SVD of H
        
        % Select eigenvectors corresponding to SVs below threshold sigma
        if d_ksum == 1
            B = B_per(d_ksum, d_ksum);
        else
            B = diag(B_per)';
        end
        
        [row1, col1] = find(0 == B); % Find cols with SV < threshold
        col = [col1 d_ksum+1:M_R];
        V_tilda1 = V_per(:, col); % Pick corresponding cols of V
        P_Real = V_tilda1 * V_tilda1';
        
        % Calculate interference for perfect CSI with NSP
        frob_per_temp(x) = 0;
        for z = 1:n
            interf_per = (H_kR(:, (z-1)*M_R+1:z*M_R)) * P_Real;
            frob_per_temp(x) = frob_per_temp(x) + norm(interf_per, 'fro');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SSVSP for perfect CSI
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ss1 = length(B);
        colms = [ss1 d_ksum+1:M_R];
        V_tilda1ms = V_per(:, colms); % Pick corresponding cols of V
        P_R1ms = V_tilda1ms * V_tilda1ms';
        
        % Calculate interference for perfect CSI with SSVSP
        frob_perms_temp(x) = 0;
        for zs = 1:n
            interf_perms = (H_kR(:, (zs-1)*M_R+1:zs*M_R)) * P_R1ms;
            frob_perms_temp(x) = frob_perms_temp(x) + norm(interf_perms, 'fro');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Channel estimation and precoding with estimated CSI
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for r = 1:length(L_t)
            % Training symbol generation
            S = (1/sqrt(2)) * (hadamard(L_t(r)) + 1i * hadamard(L_t(r)));
            S = S(1:d_ksum, :);
            
            % AWGN generation
            W = sqrt(noise_v) * (randn(M_R, L_t(r)) + 1i * randn(M_R, L_t(r)));
            
            % Signal received at radar for the whole training period
            Y = zeros(M_R, L_t(r));
            for b = 1:L_t(r)
                Y(:, b) = sqrt(rho_ratio/d_ksum) * H_bar * S(:, b) + W(:, b);
            end
            
            % ML estimation of composite channel
            H_barest = sqrt(d_ksum/rho_ratio) * Y * S' * inv(S * S');
            H_est = H_barest';
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Null-space computation for estimated CSI (NSP)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [U_es, A_es, V_es] = svd(H_est); % SVD of H
            
            % Select eigenvectors corresponding to SVs below threshold sigma
            if d_ksum == 1
                A = A_es(d_ksum, d_ksum);
            else
                A = diag(A_es)';
            end
            
            [row2, col2] = find(0 == A); % Find cols with SV < threshold
            co2 = [col2 d_ksum+1:M_R];
            V_tilda2 = V_es(:, co2); % Pick corresponding cols of V
            P_Rest = V_tilda2 * V_tilda2';
            
            % Calculate interference for estimated CSI with NSP
            frob_wchest_temp(r, x) = 0;
            for z = 1:n
                interf_wchest = (H_kR(:, (z-1)*M_R+1:z*M_R)) * P_Rest;
                frob_wchest_temp(r, x) = frob_wchest_temp(r, x) + norm(interf_wchest, 'fro');
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % SSVSP for estimated CSI
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ss2 = length(A);
            co2ms = [ss2 d_ksum+1:M_R];
            V_tilda2ms = V_es(:, co2ms); % Pick corresponding cols of V
            P_R2ms = V_tilda2ms * V_tilda2ms';
            
            % Calculate interference for estimated CSI with SSVSP
            frob_wchestms_temp(r, x) = 0;
            for zs = 1:n
                interf_wchestms = (H_kR(:, (zs-1)*M_R+1:zs*M_R)) * P_R2ms;
                frob_wchestms_temp(r, x) = frob_wchestms_temp(r, x) + norm(interf_wchestms, 'fro');
            end
        end
    end
    
    % Calculate average values for this K value
    frob_null_per(n) = mean(frob_per_temp);
    frob_ssvsp_per(n) = mean(frob_perms_temp);
    
    frob_null_wchest1(n) = mean(frob_wchest_temp(1, :));
    frob_null_wchest2(n) = mean(frob_wchest_temp(2, :));
    
    frob_ssvsp_wchest1(n) = mean(frob_wchestms_temp(1, :));
    frob_ssvsp_wchest2(n) = mean(frob_wchestms_temp(2, :));
end

% Plot results
figure(1)
plot(K, frob_null_per, '-mp', 'LineWidth', 2)
axis([1 10 -1 16])
hold on
plot(K, frob_null_wchest1, '-c+', 'LineWidth', 2)
hold on
plot(K, frob_null_wchest2, '-gd', 'LineWidth', 2)
hold on
plot(K, frob_ssvsp_per, '-r.', 'LineWidth', 2)
hold on
plot(K, frob_ssvsp_wchest1, '-ko', 'LineWidth', 2)
hold on
plot(K, frob_ssvsp_wchest2, '-bs', 'LineWidth', 2) % 修正了变量名

xlabel('Number of BSs per cluster ($M_i$)', 'Fontsize', 14, 'Interpreter', 'latex')
ylabel('$\sum_{m_i=1}^{M_i} ||H_{m_i,R} P_{R,i}||_F$', 'Fontsize', 14, 'Interpreter', 'latex')
legend('NSP (perfect CSI)', ...
    'NSP (estimated CSI, $L_t=64$, SNR=20dB)', ...
    'NSP (estimated CSI, $L_t=256$, SNR=20dB)', ...
    'SSVSP (perfect CSI)', ...
    'SSVSP (estimated CSI, $L_t=64$, SNR=20dB)', ...
    'SSVSP (estimated CSI, $L_t=256$, SNR=20dB)', ...
    'Interpreter', 'latex',...
    'Location', 'Best')

grid on;
title('Radar Interference vs Number of BSs per Cluster');
hold off;