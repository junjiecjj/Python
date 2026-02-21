% Figure 3.10 - BER versus SNR in cooperation mode with channel estimation
clear all;
close all;
clc;

% Parameters
K = 4; % No of transmitter-receiver pair in the comm system
ind = 1;

if ind == 1
    M_k = 6 * ones(1, K);
    N_k = 6 * ones(1, K);
    M_R = 30; % No of transmit antenna elements on RADAR
    N_R = 30; % No of receive antenna elements on RADAR
else
    M_k = 8 * ones(1, K);
    N_k = 8 * ones(1, K);
    M_R = 100; % No of transmit antenna elements on RADAR
    N_R = 100; % No of receive antenna elements on RADAR
end

L_b = 256; % block length
L_t = 128;
SNR = 20; % SNR of received radar signal from target in dB
snr = 10^(SNR/10);
rho_dB = (0:15);
rho_ratio = 10.^(rho_dB/10);
noise_v = sqrt(1./rho_ratio);
noit = 1e4;  

% Preallocate BER arrays
BER_ZF = zeros(1, length(rho_dB));
BER_MM = zeros(1, length(rho_dB));

% Main simulation loop over SNR values
for n = 1:length(rho_dB)
    errCount_BERZF = 0;
    errCount_BERMM = 0;
    
    % Degree of freedom calculation
    d_k = M_k; % DoF
    d_ksum = sum(M_k);
    
    % Monte Carlo simulation loop
    for x = 1:noit
        % Generate random data block using QPSK modulation
        tmp = round(rand(2, d_ksum));
        tmp = tmp * 2 - 1;
        input = (tmp(1, :) + 1i * tmp(2, :)) / sqrt(2); % 修正：i1 -> 1i
        u = input.';
        
        % Composite channel generation
        H_bar = zeros(M_R, d_ksum);
        for a = 1:K
            H_kR = randn(d_k(a), M_R) + 1i * randn(d_k(a), M_R); % 修正：i1 -> 1i
            if (a == 1)
                dim = 1:d_k(a);
            else
                dim = (sum(d_k(1:a-1))+1):(sum(d_k(1:a)));
            end
            H_bar(:, dim) = H_kR';
        end
        
        H_real = H_bar'; % 修正：原代码H_r改为H_real保持一致性
        
        % Training symbol generation
        S = (1/sqrt(2)) * (hadamard(L_t) + 1i * hadamard(L_t));
        S = S(1:d_ksum, :);
        
        % AWGN generation for training
        W = sqrt(noise_v(n)) * (randn(M_R, L_t) + 1i * randn(M_R, L_t));
        
        % Signal received at radar for the whole training period
        Y = zeros(M_R, L_t);
        for b = 1:L_t
            Y(:, b) = sqrt(rho_ratio(n)/d_ksum) * H_bar * S(:, b) + W(:, b); % 修正：H_bar而不是H_bar'
        end
        
        % ML estimation of composite channel
        H_barest = sqrt(d_ksum/rho_ratio(n)) * Y * S' * inv(S * S');
        H_est = H_barest'; % 修正：使用H_est表示估计的信道
        
        % ZF precoder for estimated CSI
        P_R1 = H_est' * inv(H_est * H_est');
        
        % MMSE precoder for estimated CSI
        P_R1s = H_est' * inv(H_est * H_est' + d_ksum * (1/rho_ratio(n)) .* eye(d_ksum));
        
        % AWGN generation for data transmission
        W_data = sqrt(noise_v(n)/2) * (randn(d_ksum, 1) + 1i * randn(d_ksum, 1));
        
        % ZF precoding transmission
        Y_ZF = H_real * P_R1 * u + W_data;
        EstSymbols_ZF = sign(real(Y_ZF)) + 1i * sign(imag(Y_ZF));
        EstSymbols_ZF = EstSymbols_ZF / sqrt(2);
        
        % Count errors for ZF (real part)
        I_br_ZF = find((real(u) - real(EstSymbols_ZF)) == 0);
        errCount_BERZF = errCount_BERZF + (d_ksum - length(I_br_ZF));
        
        % Count errors for ZF (imaginary part)
        I_bi_ZF = find((imag(u) - imag(EstSymbols_ZF)) == 0);
        errCount_BERZF = errCount_BERZF + (d_ksum - length(I_bi_ZF));
        
        % MMSE precoding transmission
        Y_MMSE = H_real * P_R1s * u + W_data;
        EstSymbols_MM = sign(real(Y_MMSE)) + 1i * sign(imag(Y_MMSE));
        EstSymbols_MM = EstSymbols_MM / sqrt(2);
        
        % Count errors for MMSE (real part)
        I_br = find((real(u) - real(EstSymbols_MM)) == 0);
        errCount_BERMM = errCount_BERMM + (d_ksum - length(I_br));
        
        % Count errors for MMSE (imaginary part)
        I_bi = find((imag(u) - imag(EstSymbols_MM)) == 0);
        errCount_BERMM = errCount_BERMM + (d_ksum - length(I_bi));
    end
    
    % Calculate the bit error rate (BER) for this SNR
    BER_ZF(1, n) = errCount_BERZF / (2 * d_ksum * noit);
    BER_MM(1, n) = errCount_BERMM / (2 * d_ksum * noit);
    
    % Display progress
    fprintf('SNR = %d dB: BER_ZF = %.4e, BER_MM = %.4e\n', rho_dB(n), BER_ZF(n), BER_MM(n));
end

% Plot results
figure(1)
semilogy(rho_dB, BER_ZF, '-bo', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogy(rho_dB, BER_MM, '-rs', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('SNR (dB)', 'FontSize', 14);
ylabel('Bit Error Rate (BER)', 'FontSize', 14);
legend('ZF Precoder', 'MMSE Precoder', 'Location', 'Best');
title('BER vs SNR in Cooperation Mode with Channel Estimation', 'FontSize', 14);
grid on;
hold off;

% Additional analysis: calculate performance improvement
BER_improvement = BER_ZF ./ BER_MM;
fprintf('\nAverage BER improvement (ZF/MMSE): %.2f\n', mean(BER_improvement));

% Plot BER improvement
figure(2)
plot(rho_dB, BER_improvement, '-k^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('SNR (dB)', 'FontSize', 14);
ylabel('BER Improvement (ZF/MMSE)', 'FontSize', 14);
title('BER Improvement of MMSE over ZF', 'FontSize', 14);
grid on;