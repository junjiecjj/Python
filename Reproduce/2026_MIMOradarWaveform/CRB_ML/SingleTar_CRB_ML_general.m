clc;
clear;
close all;
rng(42);

%% Parameters
M = 8;
N = 6;
L = 64;
d_lambda = 0.5;
Pt = N;
theta_true_deg = 0;
beta = 1;
SNR_dB_vec = -40:5:0;
MC_trials = 1000;

theta_search_min = theta_true_deg - 20;
theta_search_max = theta_true_deg + 20;
coarse_step = 0.02;
fine_step = 0.001;
fine_width = 0.02;

nt = (-(N - 1) / 2 : (N - 1) / 2).';
nr = (-(M - 1) / 2 : (M - 1) / 2).';

nt = (0:N-1).';
nr = (0:M-1).';

a_fun = @(theta_rad) exp(1j * 2 * pi * d_lambda * nt * sin(theta_rad));
v_fun = @(theta_rad) exp(1j * 2 * pi * d_lambda * nr * sin(theta_rad));
da_fun = @(theta_rad) 1j * 2 * pi * d_lambda * cos(theta_rad) * nt .* a_fun(theta_rad);
dv_fun = @(theta_rad) 1j * 2 * pi * d_lambda * cos(theta_rad) * nr .* v_fun(theta_rad);

theta_true_rad = theta_true_deg * pi / 180;
a0 = a_fun(theta_true_rad);

R_orth = eye(N);
R_coherent = a0 * a0';

R_orth = Pt * R_orth / real(trace(R_orth));
R_coherent = Pt * R_coherent / real(trace(R_coherent));

%% Initialization
CRB_orth_89_deg = zeros(size(SNR_dB_vec));
CRB_coherent_89_deg = zeros(size(SNR_dB_vec));
CRB_orth_90_deg = zeros(size(SNR_dB_vec));
CRB_coherent_90_deg = zeros(size(SNR_dB_vec));

RMSE_orth_deg = zeros(size(SNR_dB_vec));
RMSE_coherent_deg = zeros(size(SNR_dB_vec));

%% Monte Carlo Simulation
for idxSNR = 1:length(SNR_dB_vec)
    SNR_dB = SNR_dB_vec(idxSNR);
    SNR_lin = 10^(SNR_dB / 10);
    sigma2 = abs(beta)^2 / SNR_lin;

    CRB_orth_89_rad2 = crb_single_target_eq89(theta_true_rad, beta, R_orth, L, sigma2, a_fun, v_fun, da_fun, dv_fun);
    CRB_coherent_89_rad2 = crb_single_target_eq89(theta_true_rad, beta, R_coherent, L, sigma2, a_fun, v_fun, da_fun, dv_fun);
    CRB_orth_90_rad2 = crb_single_target_eq90(theta_true_rad, beta, R_orth, L, sigma2, a_fun, v_fun, da_fun, dv_fun);
    CRB_coherent_90_rad2 = crb_single_target_eq90(theta_true_rad, beta, R_coherent, L, sigma2, a_fun, v_fun, da_fun, dv_fun);

    CRB_orth_89_deg(idxSNR) = sqrt(max(real(CRB_orth_89_rad2), 0)) * 180 / pi;
    CRB_coherent_89_deg(idxSNR) = sqrt(max(real(CRB_coherent_89_rad2), 0)) * 180 / pi;
    CRB_orth_90_deg(idxSNR) = sqrt(max(real(CRB_orth_90_rad2), 0)) * 180 / pi;
    CRB_coherent_90_deg(idxSNR) = sqrt(max(real(CRB_coherent_90_rad2), 0)) * 180 / pi;

    theta_est_orth_deg = zeros(MC_trials, 1);
    theta_est_coherent_deg = zeros(MC_trials, 1);

    fprintf('SNR = %.1f dB\n', SNR_dB);

    for idxMC = 1:MC_trials
        X_orth = generate_waveform_exact_covariance(R_orth, L);
        X_coherent = generate_waveform_exact_covariance(R_coherent, L);

        Y_orth = simulate_single_target_matrix(theta_true_rad, beta, X_orth, sigma2, a_fun, v_fun);
        Y_coherent = simulate_single_target_matrix(theta_true_rad, beta, X_coherent, sigma2, a_fun, v_fun);

        theta_est_orth_deg(idxMC) = ml_single_target_matrix_search(Y_orth, X_orth, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width, a_fun, v_fun);
        theta_est_coherent_deg(idxMC) = ml_single_target_matrix_search(Y_coherent, X_coherent, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width, a_fun, v_fun);
    end

    RMSE_orth_deg(idxSNR) = sqrt(mean((theta_est_orth_deg - theta_true_deg).^2));
    RMSE_coherent_deg(idxSNR) = sqrt(mean((theta_est_coherent_deg - theta_true_deg).^2));
end

%% Plot
figure;
semilogy(SNR_dB_vec, CRB_coherent_89_deg, 'b-', 'LineWidth', 1.8);
hold on;
semilogy(SNR_dB_vec, CRB_orth_89_deg, 'r-', 'LineWidth', 1.8);
semilogy(SNR_dB_vec, CRB_coherent_90_deg, 'bo', 'LineWidth', 1.3, 'MarkerSize', 7);
semilogy(SNR_dB_vec, CRB_orth_90_deg, 'rd', 'LineWidth', 1.3, 'MarkerSize', 7);
semilogy(SNR_dB_vec, RMSE_coherent_deg, 'b--o', 'LineWidth', 1.3, 'MarkerSize', 7);
semilogy(SNR_dB_vec, RMSE_orth_deg, 'r--d', 'LineWidth', 1.3, 'MarkerSize', 7);
grid on;
xlabel('SNR (dB)', 'Interpreter', 'latex');
ylabel('RMSE of $\theta$ (deg)', 'Interpreter', 'latex');
legend('CRB Eq. (89), coherent signal', ...
       'CRB Eq. (89), orthogonal signal', ...
       'CRB Eq. (90), coherent signal', ...
       'CRB Eq. (90), orthogonal signal', ...
       'ML, coherent signal', ...
       'ML, orthogonal signal', ...
       'Location', 'northeast', ...
       'Interpreter', 'latex');
xlim([min(SNR_dB_vec), max(SNR_dB_vec)]);

%% Figure Style
width = 6;
height = 4;
fontsize = 14;
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
set(gcf, 'Units', 'inches');
set(gcf, 'Color', 'white');
set(gcf, 'Renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);
set(gcf, 'PaperPositionMode', 'manual');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer', 'bottom');
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.12, 0.14, 0.86, 0.82]);

%% Local Functions
function X = generate_waveform_exact_covariance(R, L)
    R = (R + R') / 2;
    [U, D] = eig(R);
    lambda = real(diag(D));
    idx = lambda > 1e-10;
    U = U(:, idx);
    lambda = lambda(idx);
    rankR = length(lambda);
    if L < rankR
        error('L must be no smaller than rank(R).');
    end
    G = randn(L, rankR) + 1j * randn(L, rankR);
    [Q, ~] = qr(G, 0);
    V = Q';
    F = U * diag(sqrt(lambda));
    X = sqrt(L) * F * V;
end

function Y = simulate_single_target_matrix(theta_rad, beta, X, sigma2, a_fun, v_fun)
    M = length(v_fun(theta_rad));
    L = size(X, 2);
    a = a_fun(theta_rad);
    v = v_fun(theta_rad);
    Z = sqrt(sigma2 / 2) * (randn(M, L) + 1j * randn(M, L));
    Y = beta * v * a' * X + Z;
end

function theta_hat_deg = ml_single_target_matrix_search(Y, X, theta_min_deg, theta_max_deg, coarse_step_deg, fine_step_deg, fine_width_deg, a_fun, v_fun)
    theta_grid_deg = theta_min_deg:coarse_step_deg:theta_max_deg;
    theta_hat_deg = ml_single_target_matrix_grid(Y, X, theta_grid_deg, a_fun, v_fun);
    theta_grid_deg = theta_hat_deg - fine_width_deg:fine_step_deg:theta_hat_deg + fine_width_deg;
    theta_grid_deg = theta_grid_deg(theta_grid_deg >= theta_min_deg);
    theta_grid_deg = theta_grid_deg(theta_grid_deg <= theta_max_deg);
    theta_hat_deg = ml_single_target_matrix_grid(Y, X, theta_grid_deg, a_fun, v_fun);
end

function theta_hat_deg = ml_single_target_matrix_grid(Y, X, theta_grid_deg, a_fun, v_fun)
    R = X * X' / size(X, 2);
    bestScore = -inf;
    theta_hat_deg = theta_grid_deg(1);
    for idxTheta = 1:length(theta_grid_deg)
        theta_rad = theta_grid_deg(idxTheta) * pi / 180;
        a = a_fun(theta_rad);
        v = v_fun(theta_rad);
        numerator = abs(v' * Y * X' * a)^2;
        denominator = real((v' * v) * (a' * R * a));
        score = numerator / denominator;
        if score > bestScore
            bestScore = score;
            theta_hat_deg = theta_grid_deg(idxTheta);
        end
    end
end

function CRB_rad2 = crb_single_target_eq89(theta_rad, beta, R, L, sigma2, a_fun, v_fun, da_fun, dv_fun)
    a = a_fun(theta_rad);
    v = v_fun(theta_rad);
    adot = da_fun(theta_rad);
    vdot = dv_fun(theta_rad);
    H = v * a';
    Hdot = vdot * a' + v * adot';
    F11 = real(trace(Hdot * R * Hdot'));
    F12 = trace(H * R * Hdot');
    F22 = real(trace(H * R * H'));
    denominator = 2 * L * abs(beta)^2 * (F11 * F22 - abs(F12)^2);
    CRB_rad2 = sigma2 * F22 / denominator;
end

function CRB_rad2 = crb_single_target_eq90(theta_rad, beta, R, L, sigma2, a_fun, v_fun, da_fun, dv_fun)
    a = a_fun(theta_rad);
    v = v_fun(theta_rad);
    adot = da_fun(theta_rad);
    vdot = dv_fun(theta_rad);
    M = length(v);
    aRa = real(a' * R * a);
    adotRadot = real(adot' * R * adot);
    aRadot = a' * R * adot;
    vdotNorm2 = real(vdot' * vdot);
    denominatorInner = aRa * vdotNorm2 + M * (adotRadot - abs(aRadot)^2 / aRa);
    CRB_rad2 = sigma2 / (2 * L * abs(beta)^2 * denominatorInner);
end
