clc;
clear;
close all;

rng(42);

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultTextFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 6);
set(0, 'DefaultFigureColor', 'w');



%% MIMO Capacity maximization
M = 6;
L = 100;
N = 4;
PT = 1;
sigma_c2 = 1;

Hc = randn(M, N) + 1j * randn(M, N);

%% IID 高斯白噪声
Sigma_Z = eye(M) * sigma_c2;

Sigma_H = Hc' * Hc;
Sigma_H = (Sigma_H + Sigma_H') / 2;

[V_h, D_h] = eig(Sigma_H);
Lambda_h = real(diag(D_h));

% [Lambda_h, idx] = sort(Lambda_h, 'descend');
% V_h = V_h(:, idx);
% Lambda_h = max(Lambda_h, 0);

Lambda_x_white = water_filling(ones(N, 1) * sigma_c2, Lambda_h, PT);
water_level_white = compute_water_level(ones(N, 1) * sigma_c2, Lambda_h, PT);

fprintf('\nWhite noise, theoretical power allocation:\n');
disp(Lambda_x_white.');
fprintf('sum power = %.4f\n', sum(Lambda_x_white));

plot_waterfilling(sigma_c2 ./ Lambda_h, Lambda_x_white, water_level_white);

Sigma_X_white = V_h * diag(Lambda_x_white) * V_h';
Sigma_X_white = (Sigma_X_white + Sigma_X_white') / 2;

C_white_mat = Hc * Sigma_X_white * Hc' + Sigma_Z;
C_white_mat = (C_white_mat + C_white_mat') / 2;

C_white_logdet = logdet_hermitian_pd(C_white_mat);
C_white_true = C_white_logdet - logdet_hermitian_pd(Sigma_Z);

fprintf('White noise logdet objective = %.8f\n', C_white_logdet);
fprintf('White noise true capacity    = %.8f\n', C_white_true);

%% Colored 噪声：白化版本
Z = randn(M, L) + 1j * randn(M, L);
Sigma_Z = Z * Z' / L;
Sigma_Z = (Sigma_Z + Sigma_Z') / 2;

A_eff = Hc' * (Sigma_Z \ Hc);
A_eff = (A_eff + A_eff') / 2;

[V_eff, D_eff] = eig(A_eff);
Lambda_eff = real(diag(D_eff));

% [Lambda_eff, idx] = sort(Lambda_eff, 'descend');
% V_eff = V_eff(:, idx);
% Lambda_eff = max(Lambda_eff, 0);

Lambda_x_colored = water_filling(ones(N, 1), Lambda_eff, PT);
water_level_colored = compute_water_level(ones(N, 1), Lambda_eff, PT);

fprintf('\nColored noise, theoretical power allocation:\n');
disp(Lambda_x_colored.');
fprintf('sum power = %.4f\n', sum(Lambda_x_colored));

plot_waterfilling(1 ./ Lambda_eff, Lambda_x_colored, water_level_colored);

Sigma_X_colored = V_eff * diag(Lambda_x_colored) * V_eff';
Sigma_X_colored = (Sigma_X_colored + Sigma_X_colored') / 2;

C_colored_mat = Hc * Sigma_X_colored * Hc' + Sigma_Z;
C_colored_mat = (C_colored_mat + C_colored_mat') / 2;

C_colored_logdet = logdet_hermitian_pd(C_colored_mat);
C_colored_true = C_colored_logdet - logdet_hermitian_pd(Sigma_Z);

fprintf('Colored noise logdet objective = %.8f\n', C_colored_logdet);
fprintf('Colored noise true capacity    = %.8f\n', C_colored_true);

%% ============================ Local functions ============================

function power_allocation = water_filling(sigma2, lambda, PT)
    N = length(lambda);
    noise_terms = sigma2(:) ./ lambda(:);
    [noise_sorted, idx_sorted] = sort(noise_terms, 'ascend');
    
    water_level = 0;
    k_active = N;
    
    for k = 1:N
        water_candidate = (PT + sum(noise_sorted(1:k))) / k;
        if k == N || water_candidate <= noise_sorted(k+1)
            water_level = water_candidate;
            k_active = k;
            break;
        end
    end
    
    power_allocation = zeros(N, 1);
    
    for i = 1:N
        if i <= k_active
            power_allocation(idx_sorted(i)) = max(0, water_level - noise_sorted(i));
        else
            power_allocation(idx_sorted(i)) = 0;
        end
    end
end

function water_level = compute_water_level(sigma2, lambda, PT)
    N = length(lambda);
    noise_terms = sigma2(:) ./ lambda(:);
    noise_sorted = sort(noise_terms, 'ascend');
    
    water_level = 0;
    
    for k = 1:N
        water_candidate = (PT + sum(noise_sorted(1:k))) / k;
        if k == N || water_candidate <= noise_sorted(k+1)
            water_level = water_candidate;
            break;
        end
    end
end

function plot_waterfilling(noise_powers, optimal_powers, water_level)
    N = length(noise_powers);
    x = 1:N;
    
    fig = figure;
    set(fig, 'Position', [100, 100, 800, 600]);
    
    Y = [noise_powers(:), optimal_powers(:)];
    b = bar(x, Y, 'stacked', 'BarWidth', 0.5);
    hold on;
    
    h_water = yline(water_level, '--', sprintf('Water level: %.3f', water_level), 'LineWidth', 2);
    
    xlabel('Channel index');
    ylabel('Power');
    legend([b(1), b(2), h_water], {'Noise level', 'Power allocation', 'Water level'}, 'Location', 'best');
    grid on;
    box on;
end

function val = logdet_hermitian_pd(A)
    A = (A + A') / 2;
    R = chol(A);
    val = 2 * sum(log(real(diag(R))));
end