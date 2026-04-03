% This Matlab script can be used to generate the ambiguity function and
% range-Doppler map comparison results in the paper:
% P. Li, M. Li, R. Liu, Q. Liu, and A. L. Swindlehurst, "MIMO-OFDM ISAC waveform design for range-Doppler sidelobe suppression," IEEE Trans. Wireless Commun., vol. 24, no. 2, pp. 1001-1015, Feb. 2025.
% P. Li, M. Li, R. Liu, Q. Liu, and A. L. Swindlehurst, "Low range-Doppler sidelobe ISAC waveform design: A low-complexity approach," IEEE Trans. Veh. Technol., vol. 74, no. 10, pp. 16544-16549, Oct. 2025.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-26

clear; clc; close all;
rng('shuffle');

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));
R = load(fullfile(root,'data','waveform_result_20260326_110413.mat'), ...
    'para', 'x_comm', 'combined', 'ISAC_almrcg', 'ISAC_mmadmm', 'radar');

% Basic parameters
Ns = R.para.Ns;
Nc = R.para.Nc;
Nt = R.para.Nt;
Nr = R.para.Nr;
fc = R.para.fc;
c0 = R.para.c0;
lambda = R.para.lambda;
deltaf = R.para.deltaf;
Tsym = R.para.Tsym;
sigma2_radar = R.para.sigma_r2;
delay_res = 1 / (R.para.bandwidth);
range_res = c0 * delay_res / 2;

% Waveforms to be compared
wave_name_list  = {'ISAC_mmadmm', 'ISAC_almrcg', 'radar', 'combined'};
wave_label_list = {'MM-ADMM', 'ALM-RCG', 'Radar-only', 'Combined'};
wave_labels = cell(numel(wave_name_list), 1);
RDM_dB_all  = cell(numel(wave_name_list), 1);
AF_dB_all   = cell(numel(wave_name_list), 1);

% Two-target setup used for visualization
tarPara = struct();
tarPara.theta_rad    = R.para.theta0_rad;
tarPara.delay_bin    = [1 5];
tarPara.doppler_bin  = [0 2];
tarPara.rcs_db       = [20 0];
tarPara.Kt = numel(tarPara.delay_bin);
tarPara.delay_s = tarPara.delay_bin * delay_res;
tarPara.range_m = tarPara.delay_bin * range_res;
tarPara.rcs     = 10.^(tarPara.rcs_db / 10);
tarPara.alpha = sqrt(tarPara.rcs .* lambda^2 ./ ((4*pi)^3 .* tarPara.range_m.^4)) ...
              .* exp(-1j * 2*pi * fc * tarPara.delay_s);

% Transmit/receive steering vectors toward the target angle
aT = exp(1j*pi*(0:Nt-1).' * sin(tarPara.theta_rad));
aR = exp(1j*pi*(0:Nr-1).' * sin(tarPara.theta_rad));

noise = sqrt(sigma2_radar/2) * (randn(Nr, Nc, Ns) + 1j*randn(Nr, Nc, Ns));
N_noise = reshape(noise, Nc * Nr, Ns);
Ynoise  = kron(eye(Nc), aR') * N_noise / Nr;

%% compute AFs and RDMs
for k = 1:numel(wave_name_list)
    name = wave_name_list{k};
    if strcmp(name, 'x_comm')
        x = R.x_comm;
    else
        x = R.(name).x;
    end
    X = reshape(x, Nc * Nt, Ns);
    Xbar = kron(eye(Nc), aT') * X;

    AF = abs(fftshift(fft(ifft(Xbar .* conj(Xbar), Nc, 1), Ns, 2), 2));
    AF = fftshift(AF, 1);
    AF = AF / max(AF(:));
    AF_dB = 10 * log10(AF);

    Yecho = zeros(Nc, Ns);
    for ii = 1:tarPara.Kt
        delay_vec   = exp(-1j * 2*pi * tarPara.delay_bin(ii)   * (0:Nc-1).' / Nc);
        doppler_vec = exp(-1j * 2*pi * tarPara.doppler_bin(ii) * (0:Ns-1).' / Ns);
        Yecho = Yecho + tarPara.alpha(ii) * (delay_vec * doppler_vec') .* Xbar;
    end
    Yecho = Yecho + Ynoise;

    Z_rdm = abs(fftshift(fft(ifft(Yecho .* conj(Xbar), Nc, 1), Ns, 2), 2));
    RDM = fftshift(Z_rdm, 1);
    RDM = RDM / max(RDM(:));
    RDM_dB = 10 * log10(RDM);

    wave_labels{k} = wave_label_list{k};
    AF_dB_all{k}   = AF_dB;
    RDM_dB_all{k}  = RDM_dB;
end

%% Plot
doppler_axis = -floor(Ns/2) : (ceil(Ns/2)-1);
range_axis   = 0 : Nc-1;
delay_axis   = -floor(Nc/2) : (ceil(Nc/2)-1);

rdm_zmin = 0;
for k = 1:numel(RDM_dB_all)
    rdm_zmin = min(rdm_zmin, min(RDM_dB_all{k}(:)));
end

nPlot = numel(wave_labels);
nCol = ceil(sqrt(nPlot));
nRow = ceil(nPlot / nCol);

% AFs
figure('Color', 'w', 'Position', [500, 150, 1000, 750]);
tiledlayout(nRow, nCol, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:nPlot
    nexttile;
    af = mesh(doppler_axis, delay_axis, AF_dB_all{k});
    colormap(jet);    af.FaceColor='interp';    shading interp;
    clim([-100, 0]);       zlim([-200, 0]);
    xlim([-Ns/2, Ns/2-1]);    ylim([-Nc/2, Nc/2-1]);
    xlabel('Doppler bin');
    ylabel('Range bin');
    zlabel('Magnitude (dB)');
end
sgtitle('Ambiguity Functions');

% RDMs
figure('Color', 'w', 'Position', [500, 150, 1000, 750]);
tiledlayout(nRow, nCol, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:nPlot
    nexttile;
    rdm = mesh(doppler_axis, delay_axis, RDM_dB_all{k});
    colormap(jet);    rdm.FaceColor='interp';    shading interp;
    clim([rdm_zmin, 0]);       zlim([rdm_zmin, 0]);
    xlim([-Ns/2, Ns/2-1]);    ylim([-Nc/2, Nc/2-1]);
    view(2);    colorbar;
    xlabel('Doppler bin');
    ylabel('Range bin');

    hold on;
    m1 = plot(tarPara.doppler_bin(1), tarPara.delay_bin(1), '^', ...
        'MarkerSize', 12, 'LineWidth', 1.5, 'Color', 'k');
    m2 = plot(tarPara.doppler_bin(2), tarPara.delay_bin(2), 'o', ...
        'MarkerSize', 12, 'LineWidth', 1.5, 'Color', 'k');
    hold off;
    legend([m1, m2], {'Strong target', 'Weak target'}, 'Fontsize', 12, 'Location','northeast');
end
sgtitle('Range-Doppler Maps');

% AF slices
zero_delay_idx   = floor(Nc/2) + 1;
zero_doppler_idx = floor(Ns/2) + 1;

figure('Color', 'w', 'Position', [650, 250, 700, 500]);
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile;
for k = 1:nPlot
    plot(ax1, delay_axis, AF_dB_all{k}(:, zero_doppler_idx), 'marker', R.para.markers{k}, ...
        'color', R.para.colors{k}, 'LineWidth', 1.5);
    hold(ax1, 'on');
end
hold(ax1, 'off');
grid(ax1, 'on');
xlabel(ax1, 'Range bin','FontSize', 12)
ylabel(ax1, 'Magnitude (dB)','FontSize', 12)
xlim(ax1, [delay_axis(1), delay_axis(end)]);

ax2 = nexttile;
for k = 1:nPlot
    plot(ax2, doppler_axis, AF_dB_all{k}(zero_delay_idx, :), 'marker', R.para.markers{k}, ...
        'color', R.para.colors{k}, 'LineWidth', 1.5);
    hold(ax2, 'on');
end
hold(ax2, 'off');
grid(ax2, 'on');
xlabel(ax2, 'Doppler bin','FontSize', 12);
ylabel(ax2, 'Magnitude (dB)','FontSize', 12);
xlim(ax2, [doppler_axis(1), doppler_axis(end)]);

lgd = legend(wave_labels, 'FontSize',12, 'NumColumns', 4);
lgd.Layout.Tile = 'north';