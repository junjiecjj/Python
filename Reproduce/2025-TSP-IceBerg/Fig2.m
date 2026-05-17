clear;
clc;
close all;

%% ========== Global Plot Settings ==========
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);
set(groot, 'defaultLineLineWidth', 2);

rng(42);

%% ========== Parameters ==========
N = 128;
L = 10;
LN = L * N;

alpha = 0.35;
span = 6;
Tsym = 1;

Order = 16;
Iter = 1000;
Mint = 100;

%% ========== 16-QAM Constellation ==========
% MATLAB Communications Toolbox is required for qammod.
%
% UnitAveragePower = true ensures:
%
%   E{|s|^2} = 1
%
sym_idx = 0 : Order - 1;
sym_idx = sym_idx(:);

Constellation = qammod(sym_idx, Order, 'UnitAveragePower', true);

kappa = mean(abs(Constellation).^4);

disp(['16-QAM kurtosis = ', num2str(kappa)]);

%% ========== Generate SRRC Pulse ==========
[p, ~, ~] = srrcFunction(alpha, L, span, Tsym);

p = p(:);
p = [p; zeros(LN - length(p), 1)];

%% ========== DFT Matrices ==========
FLN = dftmtx(LN) / sqrt(LN);
FN = dftmtx(N) / sqrt(N);

%% ========== OFDM Basis ==========
% For OFDM:
%
%   U = F_N^H
%   V = U^H F_N^H = I_N
%
U = FN';
V = eye(N);
V_tilde = V .* conj(V);

%% ========== Squared Spectrum g ==========
% Eq. (23):
%
%   g = N (F_LN p) .* (F_LN^* p^*)
%
g = N * (FLN * p) .* (conj(FLN) * conj(p));
g_first = g(1:N);

%% ========== Theoretical ACF by Eq. (27) and Eq. (34) ==========
TheoAveACF_Iceberg_Eq27 = zeros(LN, 1);
TheoAveACF_OFDM_M1_Eq27 = zeros(LN, 1);
TheoAveACF_OFDM_M100_Eq27 = zeros(LN, 1);

for k = 0 : LN - 1
    fk = FLN(:, k + 1);
    fk_tilde = fk(1:N);

    phase_k = exp(-1j * 2 * pi * k / L);
    gk = g_first + (1 - g_first) * phase_k;

    r1 = L * N * abs(fk_tilde' * gk)^2;
    r2 = norm(gk)^2;
    r3 = (kappa - 2) * L * N * norm(V_tilde * (gk .* conj(fk_tilde)))^2;

    TheoAveACF_Iceberg_Eq27(k + 1) = r1;
    TheoAveACF_OFDM_M1_Eq27(k + 1) = r1 + r2 + r3;
    TheoAveACF_OFDM_M100_Eq27(k + 1) = r1 + (r2 + r3) / Mint;
end

TheoAveACF_Iceberg_Eq27 = TheoAveACF_Iceberg_Eq27 / max(TheoAveACF_Iceberg_Eq27);
TheoAveACF_OFDM_M1_Eq27 = TheoAveACF_OFDM_M1_Eq27 / max(TheoAveACF_OFDM_M1_Eq27);
TheoAveACF_OFDM_M100_Eq27 = TheoAveACF_OFDM_M100_Eq27 / max(TheoAveACF_OFDM_M100_Eq27);

TheoAveACF_Iceberg_Eq27 = TheoAveACF_Iceberg_Eq27 + 1e-10;
TheoAveACF_OFDM_M1_Eq27 = TheoAveACF_OFDM_M1_Eq27 + 1e-10;
TheoAveACF_OFDM_M100_Eq27 = TheoAveACF_OFDM_M100_Eq27 + 1e-10;

TheoAveACF_Iceberg_Eq27 = fftshift(TheoAveACF_Iceberg_Eq27);
TheoAveACF_OFDM_M1_Eq27 = fftshift(TheoAveACF_OFDM_M1_Eq27);
TheoAveACF_OFDM_M100_Eq27 = fftshift(TheoAveACF_OFDM_M100_Eq27);

%% ========== Theoretical ACF by OFDM Special Case Eq. (36) ==========
% This is the expression actually most convenient for Fig. 2 under OFDM.
%
TheoAveACF_Iceberg = zeros(LN, 1);
TheoAveACF_OFDM_M1 = zeros(LN, 1);
TheoAveACF_OFDM_M100 = zeros(LN, 1);

wave_term = sum(g_first .* (1 - g_first));

for k = 0 : LN - 1
    phase_k = exp(-1j * 2 * pi * k / L);
    gk = g_first + (1 - g_first) * phase_k;

    fk = exp(-1j * 2 * pi * k * (0:N-1).' / LN);

    r1 = abs(gk.' * conj(fk))^2;

    cosine_term = 1 - cos(2 * pi * k / L);

    r2_M1 = (kappa - 1) * ...
        (N - 2 * cosine_term * wave_term);

    r2_M100 = (kappa - 1) / Mint * ...
        (N - 2 * cosine_term * wave_term);

    TheoAveACF_Iceberg(k + 1) = r1;
    TheoAveACF_OFDM_M1(k + 1) = r1 + r2_M1;
    TheoAveACF_OFDM_M100(k + 1) = r1 + r2_M100;
end

TheoAveACF_Iceberg = TheoAveACF_Iceberg / max(TheoAveACF_Iceberg);
TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1 / max(TheoAveACF_OFDM_M1);
TheoAveACF_OFDM_M100 = TheoAveACF_OFDM_M100 / max(TheoAveACF_OFDM_M100);

TheoAveACF_Iceberg = TheoAveACF_Iceberg + 1e-10;
TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1 + 1e-10;
TheoAveACF_OFDM_M100 = TheoAveACF_OFDM_M100 + 1e-10;

TheoAveACF_Iceberg = fftshift(TheoAveACF_Iceberg);
TheoAveACF_OFDM_M1 = fftshift(TheoAveACF_OFDM_M1);
TheoAveACF_OFDM_M100 = fftshift(TheoAveACF_OFDM_M100);

%% ========== Numerical Simulation: M = 1 ==========
SimAveACF_OFDM_M1 = zeros(Iter, LN);

for k = 0 : LN - 1
    fk = FLN(:, k + 1);
    fk_tilde = fk(1:N);

    phase_k = exp(-1j * 2 * pi * k / L);
    gk = g_first + (1 - g_first) * phase_k;

    for it = 1 : Iter
        d = randi([0, Order - 1], N, 1);
        s = Constellation(d + 1);

        VHs = abs(V' * s).^2;

        Rk = sum(gk .* VHs .* conj(fk_tilde));
        SimAveACF_OFDM_M1(it, k + 1) = abs(Rk)^2;
    end
end

Sim_M1_avg = mean(SimAveACF_OFDM_M1, 1).';
Sim_M1_max = max(SimAveACF_OFDM_M1, [], 1).';
Sim_M1_min = min(SimAveACF_OFDM_M1, [], 1).';

Sim_M1_avg = Sim_M1_avg / max(Sim_M1_avg) + 1e-10;
Sim_M1_max = Sim_M1_max / max(Sim_M1_max) + 1e-10;
Sim_M1_min = Sim_M1_min / max(Sim_M1_min) + 1e-10;

Sim_M1_avg = fftshift(Sim_M1_avg);
Sim_M1_max = fftshift(Sim_M1_max);
Sim_M1_min = fftshift(Sim_M1_min);

%% ========== Numerical Simulation: M = 100 ==========
% Important:
%
% Coherent integration means:
%
%   1) average complex R_k over M slots;
%   2) then take absolute-square;
%   3) then average over Monte-Carlo iterations.
%
% Do not average |R_k|^2 over M directly.
%
SimAveACF_OFDM_M100 = zeros(Mint, Iter, LN);

for k = 0 : LN - 1
    fk = FLN(:, k + 1);
    fk_tilde = fk(1:N);

    phase_k = exp(-1j * 2 * pi * k / L);
    gk = g_first + (1 - g_first) * phase_k;

    for m = 1 : Mint
        for it = 1 : Iter
            d = randi([0, Order - 1], N, 1);
            s = Constellation(d + 1);

            VHs = abs(V' * s).^2;

            Rk = sum(gk .* VHs .* conj(fk_tilde));
            SimAveACF_OFDM_M100(m, it, k + 1) = Rk;
        end
    end
end

RkBar = squeeze(mean(SimAveACF_OFDM_M100, 1));
RkBar2 = abs(RkBar).^2;

Sim_M100_avg = mean(RkBar2, 1).';
Sim_M100_max = max(RkBar2, [], 1).';
Sim_M100_min = min(RkBar2, [], 1).';

Sim_M100_avg = Sim_M100_avg / max(Sim_M100_avg) + 1e-10;
Sim_M100_max = Sim_M100_max / max(Sim_M100_max) + 1e-10;
Sim_M100_min = Sim_M100_min / max(Sim_M100_min) + 1e-10;

Sim_M100_avg = fftshift(Sim_M100_avg);
Sim_M100_max = fftshift(Sim_M100_max);
Sim_M100_min = fftshift(Sim_M100_min);

%% ========== Plot Fig. 2 ==========
x = (-LN/2 : LN/2 - 1).';

figure;
hold on;
box on;
grid on;

fill([x; flipud(x)], ...
    [10 * log10(Sim_M1_min); flipud(10 * log10(Sim_M1_max))], ...
    [0.08, 0.66, 0.98], ...
    'FaceAlpha', 0.35, ...
    'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

fill([x; flipud(x)], ...
    [10 * log10(Sim_M100_min); flipud(10 * log10(Sim_M100_max))], ...
    [0.95, 0.45, 0.05], ...
    'FaceAlpha', 0.35, ...
    'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

plot(x, 10 * log10(TheoAveACF_OFDM_M1), ...
    'Color', [0.00, 0.45, 0.74], ...
    'LineStyle', '-', ...
    'DisplayName', 'Average Squared ACF, Theoretical');

plot(x, 10 * log10(Sim_M1_avg), ...
    'Color', [0.00, 0.65, 0.25], ...
    'LineStyle', '-', ...
    'Marker', 'o', ...
    'MarkerIndices', 1:20:LN, ...
    'MarkerSize', 8, ...
    'MarkerFaceColor', 'none', ...
    'DisplayName', 'Average Squared ACF, Numerical');

plot(x, 10 * log10(TheoAveACF_OFDM_M100), ...
    'Color', [0.85, 0.33, 0.10], ...
    'LineStyle', '-', ...
    'DisplayName', '100 Coherent Integration, Theoretical');

plot(x, 10 * log10(Sim_M100_avg), ...
    'Color', [0.93, 0.45, 0.05], ...
    'LineStyle', '-', ...
    'Marker', 'o', ...
    'MarkerIndices', 1:20:LN, ...
    'MarkerSize', 8, ...
    'MarkerFaceColor', 'none', ...
    'DisplayName', '100 Coherent Integration, Numerical');

plot(x, 10 * log10(TheoAveACF_Iceberg), ...
    'Color', 'k', ...
    'LineStyle', '--', ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Squared ACF of the Pulse ("Iceberg")');

xlabel('Delay Index');
ylabel('Ambiguity Level (dB)');

xlim([-300, 300]);
ylim([-100, 2]);

legend('Location', 'best', 'EdgeColor', 'k');

%% ========== Optional: Save Figure ==========
% saveas(gcf, 'Fig2_MATLAB.png');
% saveas(gcf, 'Fig2_MATLAB.pdf');

%% ========== Local Function ==========
function [p, t, filtDelay] = srrcFunction(beta, L, span, Tsym)

    %% ========== SRRC Pulse ==========
    % Generate square-root raised cosine pulse.
    %
    % beta : roll-off factor
    % L    : oversampling factor
    % span : filter span in symbol durations
    % Tsym : symbol duration
    %
    % Output:
    %
    %   sum |p[n]|^2 = 1
    %

    t_start = -span * Tsym / 2;
    t_end = span * Tsym / 2;
    t_step = Tsym / L;

    t = t_start : t_step : t_end;

    A = sin(pi * t * (1 - beta) / Tsym) ...
        + 4 * beta * t / Tsym .* cos(pi * t * (1 + beta) / Tsym);

    B = pi * t / Tsym .* (1 - (4 * beta * t / Tsym).^2);

    p = 1 / sqrt(Tsym) * A ./ B;

    nan_idx = isnan(p);
    inf_idx = isinf(p);

    p(nan_idx) = 1;

    p(inf_idx) = beta / sqrt(2 * Tsym) * ...
        ((1 + 2 / pi) * sin(pi / (4 * beta)) ...
        + (1 - 2 / pi) * cos(pi / (4 * beta)));

    filtDelay = (length(p) - 1) / 2;

    p = p / sqrt(sum(abs(p).^2));
    p = p(:);
end