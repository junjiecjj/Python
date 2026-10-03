%% Fig.4: original-paper RD processing
% Eq.(9) -> Eqs.(17),(20),(23) -> Eq.(29) -> Eq.(31) -> Eq.(41) -> Eq.(47)
% MATLAB R2016b or later. Run the entire file.
clear;
clc;
close all;
rng(42, 'twister');

%% 1. Paper parameters
c = 3e8;
fc = 3.5e9;
lambda = c / fc;
Br_requested = 100e6;
Deltaf = 30e3;
Tp = 1 / Deltaf;
Tcp = 8.33e-6;
Tsym = Tp + Tcp;
Ta = 2;
v = 50;
Hp = 1000;
xq = 300;
yq = 100;
Rbar = sqrt(xq^2 + Hp^2);
alpha = 1;
SNRin_dB = [-20, 5, 20];
filter_names = {'RF', 'MF', 'WF'};

%% 2. Authorized provisional settings
% Fig.4 does not explicitly specify these implementation settings.
downsample_factor = 10;
normalize_each_cut = true;
sinc_half_length = 16;
block_size = 64;
N = round(Br_requested / Deltaf);
Br = N * Deltaf;
rhor = c / (2 * Br);
Tslow = downsample_factor * Tsym;
M = round(Ta / Tslow);
m = 0:M - 1;

% Set the time origin so the target falls exactly on a slow-time sample.
mq = floor(M / 2);
y_start = yq - v * mq * Tslow;
yplatform = y_start + v * m * Tslow;

% Original n=0,...,N-1 frequency convention; signed, fftshifted Doppler.
n = (0:N - 1).';
p = -floor(M / 2):ceil(M / 2) - 1;
fD = p / (M * Tslow);
Ka = 2 * v^2 / (lambda * Rbar);

% Range cut covers the original figure's 0--4000 m interval.
k = (0:min(N - 1, floor(4000 / rhor))).';
range_axis = k * rhor;
Nr = numel(k);
[~, target_range_row] = min(abs(range_axis - Rbar));

fprintf('N=%d, Br=%.5f MHz, M=%d, aperture=%.6f s\n', N, Br / 1e6, M, M * Tslow);
fprintf('Flight interval %.3f--%.3f m; target R=%.3f m, y=%.3f m\n', yplatform(1), yplatform(end), Rbar, yq);

%% 3. Eq.(9): raw single-target echo channel
DeltaR = (yplatform - yq).^2 / (2 * Rbar);
H = alpha * exp(-1i * 4 * pi / c * ...
    ((n * Deltaf) * (Rbar + DeltaR) + fc * DeltaR));

if exist('qammod', 'file') == 2
    data = randi([0, 255], N, M);
    S = qammod(data, 256, 'UnitAveragePower', true);
    clear data;
else
    Idata = 2 * randi([0, 15], N, M) - 15;
    Qdata = 2 * randi([0, 15], N, M) - 15;
    S = (Idata + 1i * Qdata) / sqrt(170);
    clear Idata Qdata;
end
Zunit = (randn(N, M) + 1i * randn(N, M)) / sqrt(2);

%% 4. Eq.(40): reference-range migration, Eq.(47): azimuth phase filter
% A single known reference target is used for the Fig.4 calibration experiment.
Delta_k = v^2 * fD.^2 / (2 * Rbar * Ka^2 * rhor);
Haz = exp(-1i * pi * fD.^2 / Ka);

range_profiles = zeros(Nr, 3, 3);
azimuth_profiles = zeros(M, 3, 3);
fig = figure('Color', 'w', 'Position', [40, 150, 1500, 480]);

%% 5. Nine imaging runs; all filters share data and noise
for isnr = 1:3
    SNRin = 10^(SNRin_dB(isnr) / 10);
    sigma2 = abs(alpha)^2 / SNRin;
    Y = H .* S + sqrt(sigma2) * Zunit;

    for ifilter = 1:3
        switch filter_names{ifilter}
            case 'RF'                 % Eq.(17)
                Ytf = Y ./ S;
            case 'MF'                 % Eq.(20)
                Ytf = Y .* conj(S);
            case 'WF'                 % Eq.(23)
                Ytf = Y .* conj(S) ./ (abs(S).^2 + 1 / SNRin);
        end

        Yrc = sqrt(N) * ifft(Ytf, [], 1);          % Eq.(29)
        clear Ytf;
        Yrd = fftshift(fft(Yrc, [], 2), 2) / sqrt(M); % Eq.(31)
        clear Yrc;

        % Compute the requested cuts of Eq.(47) without storing a full image.
        % range_complex is the IFFT output at m=mq.
        % azimuth_spectrum is the corrected target-range row before IFFT.
        range_complex = complex(zeros(Nr, 1));
        azimuth_spectrum = complex(zeros(1, M));

        for first = 1:block_size:M
            columns = first:min(first + block_size - 1, M);
            shifts = Delta_k(columns);
            integer_shift = floor(shifts);
            fractional_shift = shifts - integer_shift;
            Yrcmc_block = complex(zeros(Nr, numel(columns)));

            for offset = -sinc_half_length:sinc_half_length
                input_k = k + integer_shift + offset;
                valid = input_k >= 0 & input_k <= N - 1;
                safe_k = min(max(input_k, 0), N - 1);
                linear_index = safe_k + 1 + N * (columns - 1);
                samples = Yrd(linear_index);
                samples(~valid) = 0;
                weights = sinc_local(offset - fractional_shift);
                Yrcmc_block = Yrcmc_block + samples .* weights; % Eq.(41)
            end

            corrected = Yrcmc_block .* Haz(columns);           % Eq.(47)
            range_complex = range_complex + corrected * exp(1i * 2 * pi * mq * p(columns) / M).' / sqrt(M);
            azimuth_spectrum(columns) = corrected(target_range_row, :);
        end
        clear Yrd;

        azimuth_complex = sqrt(M) * ifft(ifftshift(azimuth_spectrum, 2), [], 2);         % Eq.(47)
        range_cut = abs(range_complex);
        azimuth_cut = abs(azimuth_complex).';

        if normalize_each_cut
            range_reference = max(range_cut);
            azimuth_reference = max(azimuth_cut);
        else
            range_reference = abs(alpha) * sqrt(N * M);
            azimuth_reference = range_reference;
        end
        range_profiles(:, ifilter, isnr) = 20 * log10(max(range_cut / max(range_reference, eps), eps));
        azimuth_profiles(:, ifilter, isnr) = 20 * log10(max(azimuth_cut / max(azimuth_reference, eps), eps));

        [~, ir] = max(range_cut);
        [~, im] = max(azimuth_cut);
        fprintf('SNR=%g dB, %s: peak R=%.3f m, y=%.3f m\n', ...
            SNRin_dB(isnr), filter_names{ifilter}, range_axis(ir), yplatform(im));
    end

    %% 6. Original color order and six-panel arrangement
    if isnr == 1
        plot_order = [1, 2, 3];       % RF / MF / WF
        lower_limit = -50;
    elseif isnr == 2
        plot_order = [2, 1, 3];       % MF / RF / WF
        lower_limit = -80;
    else
        plot_order = [2, 1, 3];
        lower_limit = -90;
    end
    colors = [0, 0.4470, 0.7410; 0.8500, 0.3250, 0.0980; 0.9290, 0.6940, 0.1250];
    axr = subplot(1, 6, 2 * isnr - 1, 'Parent', fig);
    axa = subplot(1, 6, 2 * isnr, 'Parent', fig);
    hold(axr, 'on');
    hold(axa, 'on');
    for j = 1:3
        index = plot_order(j);
        plot(axr, range_axis, range_profiles(:, index, isnr), 'Color', colors(j, :), 'LineWidth', 0.7);
        plot(axa, yplatform, azimuth_profiles(:, index, isnr), 'Color', colors(j, :), 'LineWidth', 0.7);
    end
    xlim(axr, [0, 4000]);
    xlim(axa, [yplatform(1), yplatform(end)]);
    ylim(axr, [lower_limit, 0]);
    ylim(axa, [lower_limit, 0]);
    xlabel(axr, 'Range (m)');
    xlabel(axa, 'Azimuth (m)');
    ylabel(axr, 'Normalized magnitude (dB)');
    ylabel(axa, 'Normalized magnitude (dB)');
    title(axr, sprintf('SNR_{in}=%g dB', SNRin_dB(isnr)));
    legend(axr, filter_names(plot_order), 'Location', 'northeast');
    legend(axa, filter_names(plot_order), 'Location', 'northeast');
    set([axr, axa], 'FontName', 'Times New Roman', 'FontSize', 10);
    grid(axr, 'on');
    grid(axa, 'on');
    box(axr, 'on');
    box(axa, 'on');
    drawnow;
end

function result = sinc_local(argument)
    result = ones(size(argument));
    nonzero = abs(argument) > 1e-12;
    result(nonzero) = sin(pi * argument(nonzero)) ./ (pi * argument(nonzero));
end