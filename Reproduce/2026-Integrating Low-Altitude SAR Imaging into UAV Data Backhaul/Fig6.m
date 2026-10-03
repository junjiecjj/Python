%% Fig.6 revised: boundary, exact range resampling, and noise-model audit
% Original processing: Eq.(9), MF Eq.(20), range IFFT Eq.(29),
% azimuth FFT Eq.(31), RCMC Eqs.(40)-(41), azimuth IFFT Eq.(47).
% Supplements: seed, pilot phase alphabet/comb offset, aperture origin,
% exact periodic range interpolation and per-profile peak normalization.
% Supplemental diagnostic mode: white noise on inactive pilot subcarriers.
% This mode is NOT identical to applying MF to a physically masked grid.
% No antenna beampattern: the paper does not give its parameters.
% MATLAB R2016b or later; qammod is optional.

clear;
clc;
close all;
rng(42, 'twister');

%% 1. Common parameters
P.c = 3e8;
P.fc = 3.5e9;
P.lambda = P.c / P.fc;
P.Deltaf = 30e3;
P.Tsym = 1 / P.Deltaf + 8.33e-6;
P.Ta = 2;
P.v = 50;
P.Hp = 1000;
P.N = round(100e6 / P.Deltaf);
P.rhor = P.c / (2 * P.N * P.Deltaf);
P.xq = 300;
P.yq = 100;
P.Rbar = sqrt(P.xq^2 + P.Hp^2);
P.alpha = 1;
P.SNRin = 10^(5 / 10);
P.sigma2 = abs(P.alpha)^2 / P.SNRin;
P.block_size = 64;
P.azimuth_padding_factor = 2; % Extend output period; add no observations.
P.plot_oversampling = 8;      % Display interpolation only.
% 'masked_physical': Eq.(20), noise only on occupied pilot subcarriers.
% 'paper_white_approx': additional white noise on inactive pilot rows.
% The latter is a hypothesis suggested by the paper's flat backgrounds.
P.noise_mode = 'paper_white_approx';

% Subcarrier indices are zero-based; MATLAB row = index + 1.
pilot_n = (1667:4:1954).';
active_n = {pilot_n, pilot_n, (0:P.N - 1).'};
symbol_steps = [20 * 14, 2 * 14, 10];
method_names = {'Pilot-only, periodicity of 20 slots', ...
    'Pilot-only, periodicity of 2 slots', 'Data-aided imaging'};
colors = [0, 0.4470, 0.7410; 0.8500, 0.3250, 0.0980; ...
    0.9290, 0.6940, 0.1250];

%% 2. Figure and zoomed profiles
fig = figure('Color', 'w', 'Position', [100, 100, 1050, 620]);
axR = subplot(1, 2, 1, 'Parent', fig);
axA = subplot(1, 2, 2, 'Parent', fig);
prepare_axis(axR, 'Range (m)', [0, 5000], [-100, 0]);
prepare_axis(axA, 'Azimuth (m)', [0, 200], [-100, 0]);

zoomR = axes('Parent', fig, 'Position', [0.19, 0.30, 0.24, 0.14]);
prepare_axis(zoomR, 'Range (m)', [900, 1200], [-80, 0]);
zoomA = axes('Parent', fig, 'Position', [0.65, 0.30, 0.24, 0.14]);
prepare_axis(zoomA, 'Azimuth (m)', [95, 105], [-80, 0]);
set([zoomR, zoomA], 'FontSize', 8);

range_profiles = cell(1, 3);
azimuth_profiles = cell(1, 3);
range_axes = cell(1, 3);
azimuth_axes = cell(1, 3);

%% 3. Independently generate and image all three sampling schemes
for scheme = 1:3
    fprintf('\n%s\n', method_names{scheme});
    [Raxis, Yaxis, range_cut, azimuth_cut, Rfine, Yfine, range_fine, azimuth_fine] = ...
        image_scheme(P, active_n{scheme}, symbol_steps(scheme), scheme == 3);

    range_dB = normalized_dB(range_cut);
    azimuth_dB = normalized_dB(azimuth_cut);
    % Normalize to the native-grid peak; interpolation does not alter the reference.
    range_fine_dB = 20 * log10(max(abs(range_fine) / max(abs(range_cut)), eps));
    azimuth_fine_dB = 20 * log10(max(abs(azimuth_fine) / max(abs(azimuth_cut)), eps));
    range_profiles{scheme} = range_dB;
    azimuth_profiles{scheme} = azimuth_dB;
    range_axes{scheme} = Raxis;
    azimuth_axes{scheme} = Yaxis;

    plot(axR, Raxis, range_dB, 'Color', colors(scheme, :), ...
        'LineWidth', 1, 'DisplayName', method_names{scheme});
    plot(axA, Yfine, azimuth_fine_dB, 'Color', colors(scheme, :), ...
        'LineWidth', 1, 'DisplayName', method_names{scheme});
    plot(zoomR, Rfine, range_fine_dB, 'Color', colors(scheme, :), 'LineWidth', 1);
    plot(zoomA, Yfine, azimuth_fine_dB, 'Color', colors(scheme, :), 'LineWidth', 1);
    drawnow;

    % Local peak check avoids mistaking a periodic ambiguity for the target.
    rwindow = find(abs(Raxis - P.Rbar) < 20);
    awindow = find(abs(Yaxis - P.yq) < 5);
    [~, ir] = max(abs(range_cut(rwindow)));
    [~, ia] = max(abs(azimuth_cut(awindow)));
    fprintf('Local range peak: %.3f m; local azimuth peak: %.3f m\n', ...
        Raxis(rwindow(ir)), Yaxis(awindow(ia)));
    background = true(size(Raxis));
    for ambiguity = 0:3
        background = background & abs(Raxis - (P.Rbar + 1250 * ambiguity)) > 150;
    end
    fprintf('Range background median (includes sidelobes): %.2f dB\n', ...
        median(range_dB(background)));
end

legend(axR, 'Location', 'southwest', 'FontSize', 9);
title(axR, 'Range profile');
title(axA, 'Azimuth profile');
drawnow;

fprintf('\nComb ambiguity spacing = %.3f m\n', P.c / (2 * 4 * P.Deltaf));
fprintf('Unambiguous range = %.3f m\n', P.c / (2 * P.Deltaf));
fprintf('Expected pilot ambiguity ranges: %.3f, %.3f, %.3f, %.3f m\n', ...
    P.Rbar + (0:3) * P.c / (2 * 4 * P.Deltaf));
fprintf('Azimuth output interval is enlarged by zero-padding, not longer observation.\n');
fprintf('Noise mode: %s\n', P.noise_mode);
fprintf('No extrapolated data or manually inserted ambiguity peaks.\n');

%% Local functions
function [Raxis, Yaxis, range_cut, azimuth_cut, Rfine, Yfine, range_fine, azimuth_fine] = ...
        image_scheme(P, active_n, symbol_step, is_data)

    N = P.N;
    Tslow = symbol_step * P.Tsym;
    m = 0:floor((P.Ta - eps(P.Ta)) / Tslow);
    M = numel(m);
    Lfft = P.azimuth_padding_factor * M;
    mq = floor(Lfft / 2);
    input_columns = mq + (0:M - 1) - floor(M / 2) + 1;
    y_observed = P.yq + P.v * ((0:M - 1) - floor(M / 2)) * Tslow;
    Yaxis = P.yq + P.v * ((0:Lfft - 1) - mq) * Tslow;
    p = -floor(Lfft / 2):ceil(Lfft / 2) - 1;
    fD = p / (Lfft * Tslow);
    n = active_n(:);
    K = numel(n);

    fprintf('Active subcarriers = %d, observed symbols = %d, FFT length = %d\n', ...
        K, M, Lfft);
    fprintf('PRF = %.3f Hz; observed y = %.3f to %.3f m\n', ...
        1 / Tslow, y_observed(1), y_observed(end));

    Yrc = complex(zeros(N, Lfft));
    inactive = true(N, 1);
    inactive(n + 1) = false;

    for first = 1:P.block_size:M
        columns = first:min(first + P.block_size - 1, M);
        Mb = numel(columns);
        DeltaR = (y_observed(columns) - P.yq).^2 / (2 * P.Rbar);
        H = P.alpha * exp(-1i * 4 * pi / P.c * ...
            ((n * P.Deltaf) * (P.Rbar + DeltaR) + P.fc * DeltaR));

        if is_data
            if exist('qammod', 'file') == 2
                S = qammod(randi([0, 255], K, Mb), 256, ...
                    'UnitAveragePower', true);
            else
                Idata = 2 * randi([0, 15], K, Mb) - 15;
                Qdata = 2 * randi([0, 15], K, Mb) - 15;
                S = (Idata + 1i * Qdata) / sqrt(170);
            end
        else
            S = exp(1i * pi / 2 * randi([0, 3], K, Mb));
        end

        Z = sqrt(P.sigma2 / 2) * (randn(K, Mb) + 1i * randn(K, Mb));
        Y = H .* S + Z;
        Ytf = complex(zeros(N, Mb));
        Ytf(n + 1, :) = Y .* conj(S); % Original MF on occupied rows.

        if ~is_data && strcmp(P.noise_mode, 'paper_white_approx')
            % EXPLICIT SUPPLEMENT: retain white noise on the inactive rows.
            % This tests a homogeneous noise approximation; it is not masked MF.
            Ytf(inactive, :) = sqrt(P.sigma2 / 2) * ...
                (randn(nnz(inactive), Mb) + 1i * randn(nnz(inactive), Mb));
        elseif ~strcmp(P.noise_mode, 'masked_physical') && ...
                ~strcmp(P.noise_mode, 'paper_white_approx')
            error('Unknown noise mode.');
        end

        Yrc(:, input_columns(columns)) = sqrt(N) * ifft(Ytf, [], 1);
    end

    Yrd = fftshift(fft(Yrc, [], 2), 2) / sqrt(Lfft);
    clear Yrc;

    Ka = 2 * P.v^2 / (P.lambda * P.Rbar);
    Delta_k = P.v^2 * fD.^2 / (2 * P.Rbar * Ka^2 * P.rhor);
    Haz = exp(-1i * pi * fD.^2 / Ka);
    k = (0:N - 1).';
    Raxis = k * P.rhor;
    [~, target_row] = min(abs(Raxis - P.Rbar));
    range_cut = complex(zeros(N, 1));
    azimuth_spectrum = complex(zeros(1, Lfft));

    for first = 1:P.block_size:Lfft
        columns = first:min(first + P.block_size - 1, Lfft);

        % Exact evaluation of the periodic N-point range Fourier series at
        % k+Delta_k. This implements resampling without a truncated sinc kernel.
        % The uncentered n=0,...,N-1 convention matches Eq.(9).
        range_spectrum = fft(Yrd(:, columns), [], 1) / sqrt(N);
        shifted = sqrt(N) * ifft(range_spectrum .* ...
            exp(1i * 2 * pi / N * k * Delta_k(columns)), [], 1);
        corrected = shifted .* Haz(columns);

        range_cut = range_cut + corrected * ...
            exp(1i * 2 * pi * mq * p(columns) / Lfft).' / sqrt(Lfft);
        azimuth_spectrum(columns) = corrected(target_row, :);
    end

    azimuth_cut = sqrt(Lfft) * ifft(ifftshift(azimuth_spectrum, 2), [], 2);

    % Display interpolation: preserve native-grid samples and phase conventions.
    os = P.plot_oversampling;
    range_fine = os * ifft([fft(range_cut); zeros((os - 1) * N, 1)]);
    Rfine = (0:os * N - 1).' * P.rhor / os;
    azimuth_fine = interpft(azimuth_cut, os * Lfft, 2);
    Yfine = Yaxis(1) + (0:os * Lfft - 1) * P.v * Tslow / os;

end

function values = normalized_dB(profile)
    magnitude = abs(profile);
    values = 20 * log10(max(magnitude / max(max(magnitude), eps), eps));
end

function prepare_axis(ax, label, xlimits, ylimits)
    hold(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 0.8);
    xlabel(ax, label);
    ylabel(ax, 'Normalized Magnitude (dB)');
    xlim(ax, xlimits);
    ylim(ax, ylimits);
    grid(ax, 'on');
    box(ax, 'on');
end
