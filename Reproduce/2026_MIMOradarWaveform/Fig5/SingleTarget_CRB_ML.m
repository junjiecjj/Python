clc;
clear;
close all;

addpath('./functions');
rng(42);
%% Parameters
M = 10;
L = 1;
d_lambda = 0.5;
theta_true_deg = 0;
alpha = 1;
SNR_dB_vec = -30:5:10;
MC_trials = 1000;
N_coherent = 1;
N_orth = M;
beta_coherent = 0.5;
beta_orth = 0;
theta_search_min = theta_true_deg-20;
theta_search_max = theta_true_deg+20;
coarse_step = 0.01;
fine_step = 0.001;
fine_width = 0.01;
%   n = (0:M-1).';
n = (-(M-1)/2 : (M-1)/2).';
a_fun = @(theta_deg) exp(-1j * 2 * pi * d_lambda * n * sind(theta_deg));
da_fun = @(theta_deg) -1j * 2 * pi * d_lambda * cosd(theta_deg) * n .* a_fun(theta_deg);
A_fun = @(theta_deg) a_fun(theta_deg) * a_fun(theta_deg)';
dA_fun = @(theta_deg) da_fun(theta_deg) * a_fun(theta_deg)' + a_fun(theta_deg) * da_fun(theta_deg)';
Rs_fun = @(beta) (1 - beta) * eye(M) + beta * ones(M);

%% CRB and ML
CRB_coherent_deg = zeros(size(SNR_dB_vec));
CRB_orth_deg = zeros(size(SNR_dB_vec));
CRB_coherent_63_deg = zeros(size(SNR_dB_vec));
CRB_orth_63_deg = zeros(size(SNR_dB_vec));
CRB_coherent_67_deg = zeros(size(SNR_dB_vec));
CRB_orth_67_deg = zeros(size(SNR_dB_vec));

RMSE_coherent_deg = zeros(size(SNR_dB_vec));
RMSE_orth_deg = zeros(size(SNR_dB_vec));
for idx = 1:length(SNR_dB_vec)
    SNR_dB = SNR_dB_vec(idx);
    SNR_lin = 10^(SNR_dB / 10);
    sigma2 = abs(alpha)^2 / SNR_lin;

    a0 = a_fun(theta_true_deg);
    adot0 = da_fun(theta_true_deg);
    Rs_coherent =  a_fun(theta_true_deg) * a_fun(theta_true_deg)';  %  Rs_fun(beta_coherent);
    Rs_orth = Rs_fun(beta_orth);

    % Eq.(57-59)
    CRB_coherent_rad2 = crb_single_target_equiv_model(theta_true_deg, alpha, Rs_coherent, M, N_coherent, sigma2, A_fun, dA_fun);
    CRB_orth_rad2 = crb_single_target_equiv_model(theta_true_deg, alpha, Rs_orth, M, N_orth, sigma2, A_fun, dA_fun);
    CRB_coherent_deg(idx) = sqrt(max(real(CRB_coherent_rad2), 0)) * 180 / pi;
    CRB_orth_deg(idx) = sqrt(max(real(CRB_orth_rad2), 0)) * 180 / pi;
    
    SNR_coherent_eff = N_coherent * abs(alpha)^2 / sigma2;
    SNR_orth_eff = N_orth * abs(alpha)^2 / sigma2; 

    CRB_orth_63_rad2 = crb_single_63(a0, adot0, Rs_orth, SNR_orth_eff);
    CRB_orth_63_deg(idx) = sqrt(max(real(CRB_orth_63_rad2), 0)) * 180 / pi;
    CRB_orth_67_rad2 = crb_single_67_correct(a0, adot0, Rs_orth, SNR_orth_eff); 
    CRB_orth_67_deg(idx) = sqrt(max(real(CRB_orth_67_rad2), 0)) * 180 / pi;

    CRB_coherent_63_rad2 = crb_single_63(a0, adot0, Rs_coherent, SNR_coherent_eff);
    CRB_coherent_63_deg(idx) = sqrt(max(real(CRB_coherent_63_rad2), 0)) * 180 / pi;
    CRB_coherent_67_rad2 = crb_single_67_correct(a0, adot0, Rs_coherent, SNR_coherent_eff);
    CRB_coherent_67_deg(idx) = sqrt(max(real(CRB_coherent_67_rad2), 0)) * 180 / pi;

    theta_est_coherent = zeros(MC_trials, 1);
    theta_est_orth = zeros(MC_trials, 1);
    fprintf('SNR = %.1f dB\n', SNR_dB);
    for mc = 1:MC_trials
        z_coherent = simulate_single_target_equiv(theta_true_deg, alpha, Rs_coherent, M, N_coherent, sigma2, A_fun);
        z_orth = simulate_single_target_equiv(theta_true_deg, alpha, Rs_orth, M, N_orth, sigma2, A_fun);
        theta_est_coherent(mc) = ml_single_target_search(z_coherent, Rs_coherent, M, N_coherent, A_fun, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width);
        theta_est_orth(mc) = ml_single_target_search(z_orth, Rs_orth, M, N_orth, A_fun, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width);
    end
    RMSE_coherent_deg(idx) = sqrt(mean((theta_est_coherent - theta_true_deg).^2));
    RMSE_orth_deg(idx) = sqrt(mean((theta_est_orth - theta_true_deg).^2));
end
%% Plot
figure(1);
semilogy(SNR_dB_vec, CRB_coherent_deg, 'b-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, CRB_orth_deg, 'r-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, RMSE_coherent_deg, 'b--', 'MarkerSize', 7, 'LineWidth', 1); hold on;
semilogy(SNR_dB_vec, RMSE_orth_deg, 'r--', 'MarkerSize', 7, 'LineWidth', 1);
xlabel('SNR [dB]');
ylabel('RMSE of \theta [deg]');
legend('CRB, \beta = 1', 'CRB, \beta = 0', 'ML, \beta = 1', 'ML, \beta = 0', 'Location', 'southwest');
grid on;
grid minor;
xlim([min(SNR_dB_vec), max(SNR_dB_vec)]);
title('Single-target CRB and ML performance versus SNR');

figure(2);
semilogy(SNR_dB_vec, CRB_orth_deg, 'k-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, CRB_orth_63_deg, 'ro', 'MarkerSize', 7, 'LineWidth', 1.3); hold on;
semilogy(SNR_dB_vec, CRB_orth_67_deg, 'b^', 'MarkerSize', 7, 'LineWidth', 1.3);
xlabel('SNR [dB]');
ylabel('RMSE CRB of \theta [deg]');
legend('Equivalent FIM', 'Formula (63)', 'Corrected Formula (67)', 'Location', 'southwest');
grid on;
grid minor;
xlim([min(SNR_dB_vec), max(SNR_dB_vec)]);
title('Orthogonal signal: comparison of three CRB calculations');
%% CRB comparison, coherent
figure(3);
semilogy(SNR_dB_vec, CRB_coherent_deg, 'k-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, CRB_coherent_63_deg, 'ro', 'MarkerSize', 7, 'LineWidth', 1.3); hold on;
semilogy(SNR_dB_vec, CRB_coherent_67_deg, 'b^', 'MarkerSize', 7, 'LineWidth', 1.3);
xlabel('SNR [dB]');
ylabel('RMSE CRB of \theta [deg]');
legend('Equivalent FIM', 'Formula (63)', 'Corrected Formula (67)', 'Location', 'southwest');
grid on;
grid minor;
xlim([min(SNR_dB_vec), max(SNR_dB_vec)]);
title('Coherent signal: comparison of three CRB calculations');







