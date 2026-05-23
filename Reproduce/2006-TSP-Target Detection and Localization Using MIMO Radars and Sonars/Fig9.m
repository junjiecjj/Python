clc;
clear;
close all;
rng(42);
%% Parameters
M = 10;
L = 2;
d_lambda = 0.5;
SNR_dB = 0;
SNR_lin = 10^(SNR_dB / 10);
N_coherent = 1;
N_orth = M;
alpha = ones(L, 1);
sigma2 = abs(alpha(1))^2 / SNR_lin;
theta1_true_deg = 0;
theta2_crb_deg = 0:0.1:3;
theta2_ml_deg = 0:0.2:3;
MC_trials = 200;
beta_coherent = 1;
beta_orth = 0;
theta_search_min = -1;
theta_search_max = 3;
coarse_step = 0.01;
fine_step = 0.001;
fine_width = 0.1;
n = (0:M-1).';
% n = (-(M - 1) / 2 : (M - 1) / 2).';
a_fun = @(theta_deg) exp(-1j * 2 * pi * d_lambda * n * sind(theta_deg));
da_fun = @(theta_deg) -1j * 2 * pi * d_lambda  * n * cosd(theta_deg) .* a_fun(theta_deg);
A_fun = @(theta_deg) a_fun(theta_deg) * a_fun(theta_deg).';
dA_fun = @(theta_deg) da_fun(theta_deg) * a_fun(theta_deg).' + a_fun(theta_deg) * da_fun(theta_deg).';
%% CRB curves
CRB_coherent_deg = zeros(size(theta2_crb_deg));
CRB_orth_deg = zeros(size(theta2_crb_deg));
for idx = 1:length(theta2_crb_deg)
    theta_true_deg = zeros(L, 1);
    theta_true_deg(1) = theta1_true_deg;
    theta_true_deg(2) = theta2_crb_deg(idx);
    CRB_coherent_rad2 = crb_theta1_equiv_model(theta_true_deg, alpha, beta_coherent, M, N_coherent, sigma2, A_fun, dA_fun);
    CRB_orth_rad2 = crb_theta1_equiv_model(theta_true_deg, alpha, beta_orth, M, N_orth, sigma2, A_fun, dA_fun);
    CRB_coherent_deg(idx) = sqrt(max(real(CRB_coherent_rad2), 0)) * 180 / pi;
    CRB_orth_deg(idx) = sqrt(max(real(CRB_orth_rad2), 0)) * 180 / pi;
end
%% ML Monte Carlo
RMSE_coherent_deg = zeros(size(theta2_ml_deg));
RMSE_orth_deg = zeros(size(theta2_ml_deg));
for idx = 1:length(theta2_ml_deg)
    theta2_true_deg = theta2_ml_deg(idx);
    theta_true_deg = zeros(L, 1);
    theta_true_deg(1) = theta1_true_deg;
    theta_true_deg(2) = theta2_true_deg;
    theta1_est_coherent = zeros(MC_trials, 1);
    theta1_est_orth = zeros(MC_trials, 1);
    fprintf('theta2 = %.2f deg\n', theta2_true_deg);
    for mc = 1:MC_trials
        z_coherent = simulate_equiv_observation(theta_true_deg, alpha, beta_coherent, M, N_coherent, sigma2, A_fun);
        z_orth = simulate_equiv_observation(theta_true_deg, alpha, beta_orth, M, N_orth, sigma2, A_fun);
        theta_hat_coherent = ml_two_target_joint_search(z_coherent, beta_coherent, M, N_coherent, A_fun, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width);
        theta_hat_orth = ml_two_target_joint_search(z_orth, beta_orth, M, N_orth, A_fun, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width);
        theta1_est_coherent(mc) = theta_hat_coherent(1);
        theta1_est_orth(mc) = theta_hat_orth(1);
    end
    RMSE_coherent_deg(idx) = sqrt(mean((theta1_est_coherent - theta1_true_deg).^2));
    RMSE_orth_deg(idx) = sqrt(mean((theta1_est_orth - theta1_true_deg).^2));
end
%% Plot
figure;
semilogy(theta2_crb_deg, CRB_coherent_deg, 'b-', 'LineWidth', 1.8);
hold on;
semilogy(theta2_crb_deg, CRB_orth_deg, 'r--', 'LineWidth', 1.8);
semilogy(theta2_ml_deg, RMSE_coherent_deg, 'bo', 'MarkerSize', 7, 'LineWidth', 1.5);
semilogy(theta2_ml_deg, RMSE_orth_deg, 'r^', 'MarkerSize', 7, 'LineWidth', 1.5);
xlabel('\theta_2 [deg]');
ylabel('RMSE of \theta_1 [deg]');
legend('CRB, \beta = 1', 'CRB, \beta = 0', 'ML, \beta = 1', 'ML, \beta = 0', 'Location', 'best');
grid on;
grid minor;
ylim([1e-3, 1e2]);
title('Figure 9: CRB and ML performance, M = 10, L = 2, SNR = 0 dB');

%% Functions
% U Lambda^(1/2)
function U_sqrtL = get_U_sqrtL_rank(beta, M)
    Rs = (1 - beta) * eye(M) + beta * ones(M);
    [U, Lambda] = eig(Rs);
    lambda = real(diag(Lambda));
    keep = lambda > 1e-10;
    U = U(:, keep);
    lambda = lambda(keep);
    U_sqrtL = U * diag(sqrt(lambda));
end

% N^(1/2) A U Lambda^(1/2)
function d = d_beta_vec(theta_deg, beta, M, N_eff, A_fun)
    U_sqrtL = get_U_sqrtL_rank(beta, M);
    X = sqrt(N_eff) * A_fun(theta_deg) * U_sqrtL;
    d = X(:);
end

function dd = d_beta_deriv_vec(theta_deg, beta, M, N_eff, dA_fun)
    U_sqrtL = get_U_sqrtL_rank(beta, M);
    X = sqrt(N_eff) * dA_fun(theta_deg) * U_sqrtL;
    dd = X(:);
end

% 根据(57)-(59)计算所有参数的CRB
function CRB_theta1_rad2 = crb_theta1_equiv_model(theta_deg, alpha, beta, M, N_eff, sigma2, A_fun, dA_fun)
    L = length(theta_deg);
    d0 = d_beta_vec(theta_deg(1), beta, M, N_eff, A_fun);
    obs_dim = length(d0);
    G = zeros(obs_dim, 3 * L);
    for k = 1:L
        dk = d_beta_vec(theta_deg(k), beta, M, N_eff, A_fun);
        ddk = d_beta_deriv_vec(theta_deg(k), beta, M, N_eff, dA_fun);
        G(:, k) = alpha(k) * ddk;
        G(:, L + k) = dk;
        G(:, 2 * L + k) = 1j * dk;
    end
    J = (2 / sigma2) * real(G' * G);
    J = (J + J.') / 2;
    if rcond(J) < 1e-12
        CRB_theta1_rad2 = NaN;
    else
        J_inv = J \ eye(3 * L);
        CRB_theta1_rad2 = J_inv(1, 1);
    end
end

%% 
function z = simulate_equiv_observation(theta_deg, alpha, beta, M, N_eff, sigma2, A_fun)
    L = length(theta_deg);
    d0 = d_beta_vec(theta_deg(1), beta, M, N_eff, A_fun);
    z = zeros(size(d0));
    for k = 1:L
        dk = d_beta_vec(theta_deg(k), beta, M, N_eff, A_fun);
        z = z + alpha(k) * dk;
    end
    w = sqrt(sigma2 / 2) * (randn(size(z)) + 1j * randn(size(z)));
    z = z + w;
end

function score = ml_concentrated_score(z, D)
    G = D' * D;
    rhs = D' * z;
    if rcond(G) > 1e-12
        amp_hat = G \ rhs;
    else
        amp_hat = pinv(G) * rhs;
    end
    score = real(rhs' * amp_hat);
end

function dict = build_dictionary(theta_grid, beta, M, N_eff, A_fun)
    d0 = d_beta_vec(theta_grid(1), beta, M, N_eff, A_fun);
    dict = zeros(length(d0), length(theta_grid));
    for k = 1:length(theta_grid)
        dict(:, k) = d_beta_vec(theta_grid(k), beta, M, N_eff, A_fun);
    end
end

function theta_hat = ml_grid_search_joint(z, beta, M, N_eff, A_fun, theta1_grid, theta2_grid)
    best_score = -inf;
    theta_hat = zeros(2, 1);
    d1_grid = build_dictionary(theta1_grid, beta, M, N_eff, A_fun);
    d2_grid = build_dictionary(theta2_grid, beta, M, N_eff, A_fun);
    for i = 1:length(theta1_grid)
        theta1 = theta1_grid(i);
        d1 = d1_grid(:, i);
        for j = 1:length(theta2_grid)
            theta2 = theta2_grid(j);
            if theta2 < theta1
                continue;
            end
            d2 = d2_grid(:, j);
            D = [d1, d2];
            score = ml_concentrated_score(z, D);
            if score > best_score
                best_score = score;
                theta_hat(1) = theta1;
                theta_hat(2) = theta2;
            end
        end
    end
end

function theta_hat = ml_two_target_joint_search(z, beta, M, N_eff, A_fun, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width)
    theta_grid = theta_search_min:coarse_step:theta_search_max;
    theta_hat = ml_grid_search_joint(z, beta, M, N_eff, A_fun, theta_grid, theta_grid);
    theta1_center = theta_hat(1);
    theta2_center = theta_hat(2);
    theta1_grid = theta1_center - fine_width:fine_step:theta1_center + fine_width;
    theta2_grid = theta2_center - fine_width:fine_step:theta2_center + fine_width;
    theta1_grid = theta1_grid(theta1_grid >= theta_search_min);
    theta1_grid = theta1_grid(theta1_grid <= theta_search_max);
    theta2_grid = theta2_grid(theta2_grid >= theta_search_min);
    theta2_grid = theta2_grid(theta2_grid <= theta_search_max);
    theta_hat = ml_grid_search_joint(z, beta, M, N_eff, A_fun, theta1_grid, theta2_grid);
    theta_hat = sort(theta_hat);
end


