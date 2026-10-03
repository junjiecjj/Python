%% Fig.5 candidate: gain compensation and range-profile NMSE
% Paper filters: RF Eq.(17), MF Eq.(20), WF Eq.(23).
% Reproduction hypotheses, not explicitly specified for Fig.5:
% 1. WF output is divided by E[chi].
% 2. Error is summed over one range profile and divided by ideal peak power.
% 3. Perfect migration/azimuth-phase correction is assumed.
% This reduced model does not simulate the full RD/RCMC processing chain.

clear;
clc;
close all;
rng(42, 'twister');

%% Parameters
Deltaf = 30e3;
Tsym = 1 / Deltaf + 8.33e-6;
N = round(100e6 / Deltaf);
Ta = 2;
downsample_factor = 10; % Supplement: adopted from Section V-C.
M = round(Ta / (downsample_factor * Tsym));
SNRin_dB = -10:1:25;
QAM_order = 256;
PSK_order = 256;
method_names = {'RF', 'MF', 'WF', 'PSK'};

% MC is performed at these SNRs; theory uses the full SNR grid.
MC_SNR_dB = -10:5:25;
Iter = 20;
% Independent range-frequency samples per MC realization.
% This does not change M. Increase to N for the full range-profile size.
N_MC = 256;
run_MC = true;

%% Unit-average-power constellation
if exist('qammod', 'file') == 2
    constellation = qammod((0:QAM_order - 1).', QAM_order, ...
        'UnitAveragePower', true);
else
    levels = -15:2:15;
    [Igrid, Qgrid] = ndgrid(levels, levels);
    constellation = (Igrid(:) + 1i * Qgrid(:)) / sqrt(170);
end
a = abs(constellation).^2;
assert(abs(mean(a) - 1) < 1e-12);

%% Theory for the compensated range profile
% At the correct azimuth location, after ideal phase/migration correction,
% average M TF estimates coherently, then perform the unitary range IFFT.
% With compensated mean gain equal to 1, independent errors have variance e/M.
% The profile has N samples and ideal peak power N; their ratio is e/M.
NMSE_theory = zeros(4, numel(SNRin_dB));
mu_WF = zeros(size(SNRin_dB));

for is = 1:numel(SNRin_dB)
    gamma = 10^(SNRin_dB(is) / 10);
    chi_WF = a ./ (a + 1 / gamma);
    g2_WF = a ./ (a + 1 / gamma).^2;
    mu = mean(chi_WF);
    mu_WF(is) = mu;

    e_RF = mean(1 ./ a) / gamma;
    e_MF = mean((a - 1).^2) + mean(a) / gamma;
    e_WF = (mean((chi_WF - mu).^2) + mean(g2_WF) / gamma) / mu^2;
    e_PSK = 1 / gamma;

    NMSE_theory(:, is) = [e_RF; e_MF; e_WF; e_PSK] / M;
end

%% Plot theory immediately
colors = [0, 0.4470, 0.7410; 0.8500, 0.3250, 0.0980; ...
          0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560];
fig = figure('Color', 'w', 'Position', [100, 150, 720, 480]);
ax = axes('Parent', fig);
hold(ax, 'on');
for imethod = 1:4
    semilogy(ax, SNRin_dB, NMSE_theory(imethod, :), ...
        'Color', colors(imethod, :), 'LineWidth', 1.5, ...
        'DisplayName', method_names{imethod});
end
set(ax, 'YScale', 'log', 'FontName', 'Times New Roman', ...
    'FontSize', 12, 'LineWidth', 0.8);
xlabel(ax, 'SNR_{in} (dB)');
ylabel(ax, 'NMSE');
xlim(ax, [-10, 25]);
ylim(ax, [1e-7, 1e-2]);
xticks(ax, -10:5:25);
grid(ax, 'on');
box(ax, 'on');
legend(ax, 'Location', 'northeast');
title(ax, 'Candidate: gain-compensated range-profile NMSE');
drawnow;

%% Monte Carlo: explicitly average M samples and form the range profile
% A reference point target is placed on range bin 0, with H=1 after
% ideal migration/phase correction. Ideal peak power is N_MC.
% A different on-grid range bin changes phase/position, not the error norm.
NMSE_MC = nan(4, numel(MC_SNR_dB));
ideal_profile = sqrt(N_MC) * ifft(ones(N_MC, 1), [], 1);
ideal_peak_power = max(abs(ideal_profile).^2);

if run_MC
    for is = 1:numel(MC_SNR_dB)
        gamma = 10^(MC_SNR_dB(is) / 10);
        mu = mean(a ./ (a + 1 / gamma));
        error_sum = zeros(4, 1);

        for it = 1:Iter
            S = constellation(randi(QAM_order, N_MC, M));
            Z = (randn(N_MC, M) + 1i * randn(N_MC, M)) / sqrt(2 * gamma);
            Y = S + Z;

            % Original filters, with explicit WF average-gain compensation.
            H_RF = Y ./ S;
            H_MF = Y .* conj(S);
            H_WF = Y .* conj(S) ./ (abs(S).^2 + 1 / gamma);
            H_WF = H_WF / mu;

            S_PSK = exp(1i * 2 * pi * randi([0, PSK_order - 1], N_MC, M) ...
                / PSK_order);
            H_PSK = (S_PSK + Z) ./ S_PSK;

            % Coherent azimuth accumulation, not power averaging.
            average_TF = [mean(H_RF, 2), mean(H_MF, 2), ...
                mean(H_WF, 2), mean(H_PSK, 2)];
            profiles = sqrt(N_MC) * ifft(average_TF, [], 1);
            errors = profiles - ideal_profile;
            error_sum = error_sum + ...
                sum(abs(errors).^2, 1).' / ideal_peak_power;
        end

        % No additional division by M here: it comes from coherent averaging.
        NMSE_MC(:, is) = error_sum / Iter;
        fprintf('Completed SNR = %+d dB\n', MC_SNR_dB(is));
        for imethod = 1:4
            semilogy(ax, MC_SNR_dB(is), NMSE_MC(imethod, is), ...
                'o', 'Color', colors(imethod, :), 'MarkerSize', 5, ...
                'LineStyle', 'none', 'HandleVisibility', 'off');
        end
        drawnow;
    end

    [~, locations] = ismember(MC_SNR_dB, SNRin_dB);
    relative_error = abs(NMSE_MC ./ NMSE_theory(:, locations) - 1);
    for imethod = 1:4
        fprintf('%s: maximum MC relative error = %.2f %%\n', ...
            method_names{imethod}, 100 * max(relative_error(imethod, :)));
    end
end

fprintf('\nN = %d, M = %d; Monte Carlo profile length = %d\n', N, M, N_MC);
fprintf('MF high-SNR floor = %.9e\n', mean((a - 1).^2) / M);
fprintf('This candidate assumes WF gain compensation and profile-based NMSE.\n');