% Figure 3.6 - CRB versus Number of antennas per BS
clear all; close all; clc;

% Parameters
K = 3; % No of BS in the cluster
M_k = 8 * ones(1, K); % No of antenna elements at kth transmitter
N_k = 8 * ones(1, K); % No of antenna elements at kth receiver
M_R = 100; % No of transmit antenna elements on RADAR
N_R = 100; % No of receive antenna elements on RADAR
r_o = 5000; % distance of target to radar array in meter
fc = 3.5e9; % frequency of operation in Hz
c = 3e8; % speed of light in m/s
lamda = c / fc;
d = 3 * lamda / 4; % inter-element spacing on radar
theta = 0; % target direction in degrees
L_b = 1024; % block length
L_t = 512; % part of block used for channel estimation, has to be power of 2 for using 'hadamard'
SNR = 15; % SNR of received radar signal from target in dB
snr = 10^(SNR/10);
DoF = 1:8;
rho_db = 20; % average SNR at the radar from common system
rho_ratio = 10.^(-rho_db/10);
noise_v = sqrt(1./rho_ratio);

% Steering vectors and their differentiations
a_t = steering_ULA(theta, M_R, fc, d);
a_r = steering_ULA(theta, N_R, fc, d);
a_tdiff = diff_steering_ULA(theta, M_R, fc, d);
a_rdiff = diff_steering_ULA(theta, N_R, fc, d);

% GENERATE DATA FROM RADAR FOR TARGET DETECTION
% Generation of pulses: Orthogonal waveforms S (M_RxL) such that (1/L)*S*S' is identity matrix
s = (1/sqrt(2)) * (hadamard(L_b) + 1i * hadamard(L_b));
s = s(1:M_R, :);
Rs = (1/L_b) * s * s';

% ESTIMATION OF COMPOSITE CHANNEL FROM COMMUNICATION RECEIVER
for n = 1:8
    d_k = n * ones(1, K); % DoF
    d_ksum = sum(d_k);
    
    for x = 1:100
        % Training symbol generation
        S = (1/sqrt(2)) * (hadamard(L_t) + 1i * hadamard(L_t)); % L_t>d_ksum
        S = S(1:d_ksum, :);
        
        % Composite channel generation
        H_bar = zeros(M_R, d_ksum);
        for a = 1:K
            H_KR = randn(d_k(a), M_R) + 1i * randn(d_k(a), M_R);
            if (a == 1)
                dim = 1:d_k(a);
            else
                dim = (sum(d_k(1:a-1))+1):(sum(d_k(1:a)));
            end
            H_bar(:, dim) = H_KR';
        end
        H_real = H_bar';
        
        % Null-space computation for perfect CSI
        [U_per, B_per, V_per] = svd(H_real); % SVD of H
        
        % Select eigenvectors corresponding to SVs below threshold sigma
        if n == 1
            B1 = B_per(n, n);
        else
            B1 = diag(B_per)';
        end
        
        [row1, col1] = find(B1 == 0); % Find cols with SV < threshold
        col = [col1 d_ksum+1:M_R];
        V_tilda = V_per(:, col); % Pick corresponding cols of V
        P_R1 = V_tilda * V_tilda';
        Rs_null_per = P_R1 * P_R1';
        cb(x) = CRB(Rs, a_t, a_tdiff, a_rdiff, N_R, snr);
        cb_per(x) = CRB(Rs_null_per, a_t, a_tdiff, a_rdiff, N_R, snr);
        
        % SSVSP
        ss1 = length(B1);
        colms = [ss1 d_ksum+1:M_R];
        V_tildalms = V_per(:, colms); % Pick corresponding cols of V
        P_Rlms = V_tildalms * V_tildalms';
        Rs_null_perms = P_Rlms * P_Rlms';
        cb_perms(x) = CRB(Rs_null_perms, a_t, a_tdiff, a_rdiff, N_R, snr);
        
        for r = 1:length(noise_v)
            % AWGN generation
            W = sqrt(noise_v(r)) * (randn(M_R, L_t) + 1i * randn(M_R, L_t));
            
            % Signal received at radar for the whole training period
            Y = zeros(M_R, L_t);
            for b = 1:L_t
                Y(:, b) = sqrt(rho_ratio(r)/d_ksum) * H_bar * S(:, b) + W(:, b);
            end
            
            % ML estimation of composite channel
            H_barest = sqrt(d_ksum/rho_ratio(r)) * Y * S' * inv(S * S');
            H_est = H_barest';
            
            % Null-space computation for estimated CSI
            [U_es, A_es, V_es] = svd(H_est); % SVD of H
            
            % Select eigenvectors corresponding to SVs below threshold sigma
            if n == 1
                A1 = A_es(n, n);
            else
                A1 = diag(A_es)';
            end
            
            [row2, col2] = find(A1 == 0); % Find cols with SV < threshold
            co2 = [col2 d_ksum+1:M_R];
            V_tilda2 = V_es(:, co2); % Pick corresponding cols of V
            P_R2 = V_tilda2 * V_tilda2';
            Rs_null_wchest = P_R2 * P_R2';
            cb_wchest(r, x) = CRB(Rs_null_wchest, a_t, a_tdiff, a_rdiff, N_R, snr);
            
            % SSVSP for estimated CSI
            ss2 = length(A1);
            co2ms = [ss2 d_ksum+1:M_R];
            V_tilda2ms = V_es(:, co2ms); % Pick corresponding cols of V
            P_R2ms = V_tilda2ms * V_tilda2ms';
            Rs_null_wchestms = P_R2ms * P_R2ms';
            cb_wchestms(r, x) = CRB(Rs_null_wchestms, a_t, a_tdiff, a_rdiff, N_R, snr);
        end
    end
    
    crb(n) = mean(cb);
    crb_null_wchest1(n) = mean(cb_wchest(1, :));
    crb_ms_wchest1s(n) = mean(cb_wchestms(1, :));
    crb_null_per(n) = mean(cb_per);
    crb_ms_per(n) = mean(cb_perms);
end

% Plot results
figure(1)
semilogy(DoF, crb, '-bs', 'LineWidth', 2);
axis([1 8 10^(-10.6) 10^(-10.30)]);
hold on;
semilogy(DoF, abs(crb_null_wchest1), '-ro', 'LineWidth', 2);
hold on;
semilogy(DoF, abs(crb_ms_wchest1s), '-kd', 'LineWidth', 2);
xlabel('Number of antennas per BS (N_{BS})', 'fontsize', 14);
ylabel('RMSE (degree)', 'fontsize', 14);
legend('orthogonal signals', ...
    'NSP (estimated CSI, SNR=15 dB)', ...
    'SSVSP (estimated CSI, SNR=15 dB)', ...
    'Location', 'Best');
hold off;