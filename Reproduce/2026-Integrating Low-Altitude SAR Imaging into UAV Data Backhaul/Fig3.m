%% Fig. 3: Data-aided OFDM-SAR imaging
% Paper: Integrating Low-Altitude SAR Imaging into UAV Data Backhaul
% Stages: WF -> range compression -> azimuth FFT -> RCMC -> azimuth compression
% Paper parameters are retained; unspecified implementation choices are marked.
% Run this entire file. Local functions require MATLAB R2016b or later.

clear;
clc;
close all;

%% 1. Parameters from the paper
c = 3e8;
fc = 3.5e9;
lambda = c / fc;
Br_requested = 100e6;
Deltaf = 30e3;
Tp = 1 / Deltaf;
Tcp = 8.33e-6;
Tsym = Tp + Tcp;
Ta = 2;
Hp = 1000;
v = 50;
SNRin_dB = -20;
SNRin = 10^(SNRin_dB / 10);
QAM_order = 256;

xq = [300, 250, 300];
yq = [100, 100, 140];
Rbar = sqrt(xq.^2 + Hp^2);
Q = numel(xq);

%% 2. Explicit implementation choices
rng(42, 'twister');                  % Seed is not specified in the paper.
N = round(Br_requested / Deltaf);   % 100 MHz / 30 kHz is not an integer.
Br = N * Deltaf;
rhor = c / (2 * Br);

% Section V-C uses one OFDM symbol out of every ten.
% Adopt that computational sampling choice here; Fig. 3 does not specify it.
downsample_factor = 10;
Tslow = downsample_factor * Tsym;
M = round(Ta / Tslow);
Ta_used = M * Tslow;
PRF = 1 / Tslow;

% A 2-second flight covers 100 m. Center it on y = 120 m so both
% y = 100 m and y = 140 m lie inside the observed aperture.
% This changes the coordinate origin, not the slant-range formula in Eq. (8).
y_center = mean([min(yq), max(yq)]);
m = 0:M - 1;
tslow = (m - (M - 1) / 2) * Tslow;
yplatform = y_center + v * tslow;

% Equal fixed scattering amplitudes for this point-target realization.
% Total reference scatterer power is one; the paper does not specify phases.
alpha = ones(1, Q) / sqrt(Q);
sigma2 = sum(abs(alpha).^2) / SNRin;

% Keep range cells covering Fig. 3. Every block still uses all N subcarriers.
k = (600:800).';
range_axis = k * rhor;
Nr = numel(k);
n = (0:N - 1).';
p = -floor(M / 2):ceil(M / 2) - 1;
fD = p / (M * Tslow);
p_display = 0:M - 1;

block_size = 64;
sinc_half_length = 16;              % Not specified in the paper.
dynamic_range_dB = 45;              % Display choice; no effect on processing.

fprintf('N = %d, effective bandwidth = %.5f MHz\n', N, Br / 1e6);
fprintf('M = %d, aperture time = %.6f s, PRF = %.3f Hz\n', M, Ta_used, PRF);
fprintf('Flight interval: %.3f to %.3f m\n', ...
    yplatform(1), yplatform(end));
fprintf('Range resolution: %.4f m\n', rhor);
fprintf('Maximum target delay: %.3f us; CP: %.3f us\n', 2 * max(max(sqrt(Rbar(:).^2 + (yplatform - yq(:)).^2))) / c * 1e6, Tcp * 1e6);

fig = figure('Color', 'w', 'Position', [60, 180, 1500, 430]);

%% 3. Eq. (9), WF in Eq. (23), range compression in Eq. (29)
Yrc_full = complex(zeros(N, M));

for first = 1:block_size:M
    columns = first:min(first + block_size - 1, M);
    Mb = numel(columns);
    H = complex(zeros(N, Mb));

    for q = 1:Q
        DeltaR = (yplatform(columns) - yq(q)).^2 / (2 * Rbar(q));
        range_phase = -4 * pi / c * (n * Deltaf) * (Rbar(q) + DeltaR);
        azimuth_phase = -4 * pi / c * fc * DeltaR;
        H = H + alpha(q) * exp(1i * (range_phase + azimuth_phase));
    end

    % Uniform square 256-QAM with unit average constellation power.
    if exist('qammod', 'file') == 2
        data = randi([0, QAM_order - 1], N, Mb);
        S = qammod(data, QAM_order, 'UnitAveragePower', true);
    else
        levels = sqrt(QAM_order);
        Idata = 2 * randi([0, levels - 1], N, Mb) - (levels - 1);
        Qdata = 2 * randi([0, levels - 1], N, Mb) - (levels - 1);
        S = (Idata + 1i * Qdata) / sqrt(2 * (QAM_order - 1) / 3);
    end

    Z = sqrt(sigma2 / 2) * (randn(N, Mb) + 1i * randn(N, Mb));
    Y = H .* S + Z;
    G = conj(S) ./ (abs(S).^2 + 1 / SNRin);
    Ytf = Y .* G;
    Yrc_block = sqrt(N) * ifft(Ytf, [], 1);
    Yrc_full(:, columns) = Yrc_block;
end

Yrc = Yrc_full(k + 1, :);

ax1 = subplot(1, 4, 1, 'Parent', fig);
show_stage(ax1, m, k, Yrc, dynamic_range_dB);
xlabel(ax1, 'Azimuth index: m');
ylabel(ax1, 'Range index: k');
title(ax1, '(a) Range compression');
drawnow;

%% 4. Azimuth FFT: Eq. (31)
Yrd_full = fftshift(fft(Yrc_full, [], 2), 2) / sqrt(M);
clear Yrc_full;
Yrd = Yrd_full(k + 1, :);

ax2 = subplot(1, 4, 2, 'Parent', fig);
show_stage(ax2, p_display, k, Yrd, dynamic_range_dB);
xlabel(ax2, 'Shifted Doppler index: p');
ylabel(ax2, 'Range index: k');
title(ax2, '(b) Azimuth FFT');
xlim(ax2, floor(M / 2) + [-400, 400]);
ylim(ax2, [650, 750]);
drawnow;

%% 5. RCMC: Eqs. (40)-(43)
% Each output range row uses its own range-dependent Ka and migration.
% Positive migration: output(k,p) samples input(k + Delta_k,p).
Ka = 2 * v^2 ./ (lambda * range_axis);
Yrcmc = complex(zeros(Nr, M));

for ip = 1:M
    Delta_k = v^2 * fD(ip)^2 ./ (2 * range_axis .* Ka.^2 * rhor);
    source_k = k + Delta_k;
    nearest_k = floor(source_k);

    for offset = -sinc_half_length:sinc_half_length
        input_k = nearest_k + offset;
        distance = source_k - input_k;
        valid = input_k >= 0 & input_k <= N - 1;

        % Eq. (9) uses n = 0,...,N-1, an uncentered frequency grid.
        % Its bandlimited interpolation kernel has a complex phase factor.
        % Equivalently: center the range spectrum, use real sinc, then remodulate.
        weights = sinc_local(distance) .* ...
            exp(1i * pi * (N - 1) / N * distance);

        rows = input_k(valid) + 1;
        Yrcmc(valid, ip) = Yrcmc(valid, ip) + Yrd_full(rows, ip) .* weights(valid);
    end
end

clear Yrd_full;

ax3 = subplot(1, 4, 3, 'Parent', fig);
show_stage(ax3, p_display, k, Yrcmc, dynamic_range_dB);
xlabel(ax3, 'Shifted Doppler index: p');
ylabel(ax3, 'Range index: k');
title(ax3, '(c) RCMC');
xlim(ax3, floor(M / 2) + [-400, 400]);
ylim(ax3, [650, 750]);
drawnow;

%% 6. Azimuth compression: Eq. (47)
Haz = exp(-1i * pi * (fD.^2) ./ Ka);
Yac = sqrt(M) * ifft(ifftshift(Yrcmc .* Haz, 2), [], 2);

ax4 = subplot(1, 4, 4, 'Parent', fig);
show_stage(ax4, yplatform, range_axis, Yac, dynamic_range_dB);
xlabel(ax4, 'Azimuth (m)');
ylabel(ax4, 'Slant range (m)');
xlim(ax4, [80, 160]);
ylim(ax4, [980, 1100]);
title(ax4, '(d) Azimuth compression');
drawnow;

%% 7. Check measured peak coordinates; do not overwrite them with target positions
fprintf('\nFocused peak checks within +/-4 m of each target:\n');

for q = 1:Q
    range_rows = find(abs(range_axis - Rbar(q)) <= 4);
    azimuth_columns = find(abs(yplatform - yq(q)) <= 4);
    local_image = abs(Yac(range_rows, azimuth_columns));
    [~, peak_index] = max(local_image(:));
    [ir, im] = ind2sub(size(local_image), peak_index);
    measured_range = range_axis(range_rows(ir));
    measured_azimuth = yplatform(azimuth_columns(im));

    fprintf(['Target %d: expected (R,y) = (%.3f, %.3f) m; ' 'measured = (%.3f, %.3f) m\n'], q, Rbar(q), yq(q), measured_range, measured_azimuth);
end

%% Local functions
function show_stage(ax, x, y, data, dynamic_range_dB)
    magnitude = abs(data);
    reference = max(magnitude(:));
    image_dB = 20 * log10(max(magnitude / max(reference, eps), eps));
    imagesc(ax, x, y, image_dB);
    set(ax, 'YDir', 'normal', 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 0.8);
    caxis(ax, [-dynamic_range_dB, 0]);
    % Display-only warm palette: low magnitude yellow, high magnitude dark red.
    % The paper does not specify its exact RGB palette.
    anchors = [1.00, 0.96, 0.20; 1.00, 0.65, 0.00; 0.75, 0.00, 0.00];
    warm_map = interp1([0, 0.55, 1], anchors, linspace(0, 1, 256));
    colormap(ax, warm_map);
    box(ax, 'on');
end

function result = sinc_local(argument)
    result = ones(size(argument));
    nonzero = abs(argument) > 1e-12;
    result(nonzero) = sin(pi * argument(nonzero)) ./ (pi * argument(nonzero));
end