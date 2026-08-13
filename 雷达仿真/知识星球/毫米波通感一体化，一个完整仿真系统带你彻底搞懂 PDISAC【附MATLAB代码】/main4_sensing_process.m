% SENSING_PROCESS  (consolidated Group 4: sensing processing)
% -------------------------------------------------------------------------
% Matched filtering + range-Doppler map formation + static-clutter removal,
% plus optional receiver / CA-CFAR+interpolation figures. The shared kernel helpers
% stay standalone: func_ca_cfar_adaptive_threshold, func_sinc_interpolation.
% Prerequisite: main3_signal_channel_model produced Data_Rx/Data_Tx (+ *_no_noise).
% Suppress figures with NO_RADAR_PLOT / NO_CFAR_PLOT.

%----------------------------%
% RD map with noise and data %
%----------------------------%
Rx_mat = squeeze(Data_Rx).';   % N_chip x N_block
Tx_mat = squeeze(Data_Tx).';   % N_chip x N_block

% FFT over all blocks at once (operates column-wise by default)
Rx_freq = fft(Rx_mat);                  % N_chip x N_block
Tx_freq = fft(Tx_mat);                  % N_chip x N_block

% Matched filter: element-wise multiply across all blocks
Data_Rx_cube_mf_freq = Rx_freq .* conj(Tx_freq); % frequency-domain MF product


% FFT across the Slow-time dimension (N_block)
Data_Rx_cube_mf_fft = fft(Data_Rx_cube_mf_freq, [], 2);
Data_Rx_cube_mf_fft(:, 1) = 0; % Remove scatters because of its v = 0
RD_map = ifft(Data_Rx_cube_mf_fft, [], 1);   % (N_chip × N_block)
RD_map_shifted = fftshift(RD_map, 2);% centre Doppler at 0

%------------------------------%
% RD map without noise and data%
%------------------------------%
Rx_mat_no_noise = squeeze(Data_Rx_no_noise).'; % N_chip x N_block
Tx_mat_no_data = squeeze(Data_Tx_no_data).';   % N_chip x N_block

Rx_freq_no_noise = fft(Rx_mat_no_noise);  % N_chip x N_block
Tx_freq_no_data = fft(Tx_mat_no_data);    % N_chip x N_block

Data_Rx_cube_mf_freq_no_noise = Rx_freq_no_noise .* conj(Tx_freq_no_data);
Data_Rx_cube_mf_fft_no_noise = fft(Data_Rx_cube_mf_freq_no_noise, [], 2);
Data_Rx_cube_mf_fft_no_noise(:, 1) = 0; % Remove scatters because of its v = 0
RD_map_no_noise = ifft(Data_Rx_cube_mf_fft_no_noise, [], 1); % (N_chip × N_block)
RD_map_no_noise_shifted = fftshift(RD_map_no_noise, 2); % centre Doppler at 0

%-----------------------------------%
% RD map with noise and without data%
%-----------------------------------%
Rx_mat_no_data = squeeze(Data_Rx_no_data).'; % N_chip x N_block
Rx_freq_no_data = fft(Rx_mat_no_data);  % N_chip x N_block

Data_Rx_cube_mf_freq_noise_no_data = Rx_freq_no_data .* conj(Tx_freq_no_data);
Data_Rx_cube_mf_fft_noise_no_data = fft(Data_Rx_cube_mf_freq_noise_no_data, [], 2);
Data_Rx_cube_mf_fft_noise_no_data(:, 1) = 0; % Remove scatterers because of its v = 0
RD_map_noise_no_data = ifft(Data_Rx_cube_mf_fft_noise_no_data, [], 1);   % (N_chip × N_block)
RD_map_noise_no_data_shifted = fftshift(RD_map_noise_no_data, 2); % centre Doppler at 0

%% Radar receiver figure (optional) ------------------------------------------
if ~exist('NO_RADAR_PLOT','var') || ~NO_RADAR_PLOT
% Visualizes the radar receiver chain across N_plot blocks.
%
%   Subplot 1 : Raw received signal  Data_Rx  (time domain, real part)
%   Subplot 2 : Matched-filter output in time domain  ifft(Data_Rx_cube_mf_freq)  (real part)
%   Subplot 3 : Absolute value of the MF output  |ifft(Data_Rx_cube_mf_freq)|



%% ── Parameters ───────────────────────────────────────────────────────────────
N_plot  = 2;                                    % number of blocks to display
N_total = N_plot * N_chip;
t_total = (0:N_total-1) * T_chip * 1e6;        % time axis in µs

%% ── Assemble continuous signals across N_plot blocks ─────────────────────────
raw_rx_all    = zeros(N_total, 1);   % real part of Data_Rx
mf_td_all     = zeros(N_total, 1);   % real part of ifft(Data_Rx_cube_mf)
mf_abs_all    = zeros(N_total, 1);   % |ifft(Data_Rx_cube_mf)|

for i = 1:N_plot
    blk = (i-1)*N_chip + 1 : i*N_chip;

    % ── Subplot 1 : raw received signal (first Rx antenna) ────────────────
    raw_rx_all(blk) = real( squeeze(Data_Rx(i, :, 1)).' );

    % ── Subplot 2 & 3 : matched-filter output in time domain ──────────────
    mf_td_i        = ifft(Data_Rx_cube_mf_freq(:, i));
    mf_td_all(blk) = real( mf_td_i );
    mf_abs_all(blk)= abs(  mf_td_i );
end

%% ── Global LaTeX interpreter ─────────────────────────────────────────────────
set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% ── Helper: light vertical block-boundary lines ──────────────────────────────
draw_block_dividers = @(ax) arrayfun(@(k) ...
    xline(ax, k*N_chip*T_chip*1e6, ':', ...
          'Color', [0.7 0.7 0.7], 'LineWidth', 2, ...
          'HandleVisibility', 'off'), ...
    1:N_plot-1);

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Position', [80 80 1400 750], 'Color', 'w');

% ── Subplot 1 : Raw received signal ──────────────────────────────────────────
ax1 = subplot(3, 1, 1);
hold on;
plot(t_total, raw_rx_all, 'LineWidth', 1.0, 'Color', [0.00 0.45 0.74]);
draw_block_dividers(ax1);
hold off;
xlim([t_total(1) t_total(end)]);
grid on;  box on;
title('Received Signal $\mathbf{Y}_{\rm sen}[t] \in \mathrm{C}^{N_{\rm ant}^{\rm rx} \times N_{\rm chip}}$', 'FontWeight', 'bold');
ylabel('Amplitude');
legend({'$\mathrm{Re}\{y(t)\}$'}, 'Location', 'southoutside', 'Orientation', 'horizontal');

% ── Subplot 2 : MF output – time domain (real part) ──────────────────────────
ax2 = subplot(3, 1, 2);
hold on;
plot(t_total, mf_td_all, 'LineWidth', 1.0, 'Color', [0.85 0.33 0.10]);
draw_block_dividers(ax2);
hold off;
xlim([t_total(1) t_total(end)]);
grid on;  box on;
title('Matched Filter Output $\mathbf{Z}_{\rm sen}[t] \in \mathrm{C}^{N_{\rm ant}^{\rm rx} \times N_{\rm chip}}$', ...
      'FontWeight', 'bold');
ylabel('Amplitude');
legend({'$\mathrm{Re}\{\mathbf{Z}_{\rm sen}[t]\}$'}, ...
       'Location', 'southoutside', 'Orientation', 'horizontal');

% ── Subplot 3 : MF output – absolute value ───────────────────────────────────
ax3 = subplot(3, 1, 3);
hold on;
plot(t_total, mf_abs_all, 'LineWidth', 1.0, 'Color', [0.47 0.67 0.19]);
draw_block_dividers(ax3);
hold off;
xlim([t_total(1) t_total(end)]);
grid on;  box on;
title('Matched Filter Output $|\mathbf{Z}_{\rm sen}[t]|$ (Magnitude, Time Domain)', ...
      'FontWeight', 'bold');
xlabel(sprintf('Time ($\\mu$s) [%d steps, $T_{\\rm prbs} = %.4f\\,\\mu$s each step]', ...
               N_plot, T_pmcw*1e6));
ylabel('Magnitude');
legend({'$|\mathbf{Z}_{\rm sen}[t]|$'}, ...
       'Location', 'southoutside', 'Orientation', 'horizontal');

% ── Link x-axes for synchronized zoom/pan ────────────────────────────────────
linkaxes([ax1 ax2 ax3], 'x');
end

%% CA-CFAR + interpolation figure (optional) ---------------------------------
if ~exist('NO_CFAR_PLOT','var') || ~NO_CFAR_PLOT
% Three-panel CA-CFAR visualisation on the Range-Doppler map:
%
%   Figure 1 │ Panel 1: RD power heatmap              (dB, parula colormap)
%            │ Panel 2: Adaptive CFAR threshold map   (dB, same style)
%            │ Panel 3: RD power + detections (□) + ground-truth targets (○)
%   Figure 2 │ Estimation accuracy vs ground truth
%            │ Panel 1: Range  scatter  (true vs estimated)
%            │ Panel 2: Velocity scatter (true vs estimated)
%            │ Panel 3: Per-target range & velocity errors (bar)
%   Figure 3 │ 1-D range and Doppler slices at each detection (≤ 6 shown)
%
% Velocity sign convention: positive = target moving away from radar
% xlim / ylim come from Region_of_interest.

%% ── Run processing chain ─────────────────────────────────────────────────────


%% ── Physical axes ────────────────────────────────────────────────────────────
range_res  = c / (2 * B);
range_axis = (0 : N_chip-1) * range_res;             % metres

PRF      = 1 / T_pmcw;
fd_axis  = (-N_block/2 : N_block/2-1) * (PRF / N_block);

% NOTE: sign negated so that positive velocity = moving away from radar
vel_axis = -(Lambda / 2) * fd_axis;                  % m/s  (flipped sign)

%% ── ROI from configuration ───────────────────────────────────────────────────
roi_range = Region_of_interest(1, :);   % [R_min  R_max]  metres
roi_vel   = Region_of_interest(2, :);   % [V_min  V_max]  m/s

range_mask = range_axis >= roi_range(1) & range_axis <= roi_range(2);

%% ── CA-CFAR detection ────────────────────────────────────────────────────────
N_guard_range   = 2;
N_guard_doppler = 2;
N_train_range   = 4;
N_train_doppler = 4;
P_fa            = 1e-3;
peak_select     = true;



[threshold_map, noise_power_avg_map, detected_positions, RD_power_map] = ...
    func_ca_cfar_adaptive_threshold( ...
        RD_map_shifted, ...
        N_guard_range, N_guard_doppler, ...
        N_train_range, N_train_doppler, ...
        P_fa, peak_select);

threshold_map(~range_mask, :) = NaN;

%% ── Ground-truth: range & radial velocity ────────────────────────────────────
truth_range = sqrt(sum((Tars_position - Tx_position).^2, 1)).';
unit_vec    = (Tars_position - Tx_position) ./ ...
              (sqrt(sum((Tars_position - Tx_position).^2, 1)) + eps);
truth_vel   = sum(Tars_vel .* unit_vec, 1).';
K_truth     = length(truth_range);

%% ── dB conversion ────────────────────────────────────────────────────────────
pow_dB  = 10 * log10(RD_power_map + eps);
thr_dB  = 10 * log10(threshold_map + eps);

peak_dB  = max(pow_dB(range_mask, :), [], 'all');
clim_sig = [peak_dB - 70,  peak_dB];      % 70 dB dynamic range

%% ── Detected positions: sinc sub-bin interpolation ──────────────────────────
K_det = size(detected_positions, 1);
if K_det > 0
    [det_range, det_vel] = func_sinc_interpolation( ...
        RD_map_shifted, detected_positions, range_axis, vel_axis);
end

%% ── Association: nearest detection to each truth target ─────────────────────
% Normalise both dimensions by the ROI span so range and velocity errors are
% on a comparable scale when computing the Euclidean association distance.
assoc     = nan(K_truth, 1);   % assoc(j) = detection index for truth target j
err_R_vec = nan(K_truth, 1);   % range  error  [m]
err_v_vec = nan(K_truth, 1);   % velocity error [m/s]

if K_det > 0
    range_span = max(diff(roi_range), eps);
    vel_span   = max(diff(roi_vel),   eps);
    for j = 1 : K_truth
        dist2 = ((det_range - truth_range(j)) / range_span).^2 + ...
                ((det_vel   - truth_vel(j)  ) / vel_span  ).^2;
        [~, assoc(j)] = min(dist2);
        err_R_vec(j)  = det_range(assoc(j)) - truth_range(j);
        err_v_vec(j)  = det_vel(assoc(j))   - truth_vel(j);
    end
end

%% ── Shared axis styling ──────────────────────────────────────────────────────

%% ── Global LaTeX interpreter ─────────────────────────────────────────────────
set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% ════════════════════════════════════════════════════════════════════════════
%  FIGURE 1 — Three-panel RD map
%% ════════════════════════════════════════════════════════════════════════════
fig1 = figure('Position', [60 80 1620 540], 'Color', 'w');
ax   = gobjects(1, 3);

% Panel 1: RD power heatmap ──────────────────────────────────────────────────
ax(1) = subplot(1, 3, 1);
imagesc(vel_axis, range_axis, pow_dB);
clim(clim_sig);
style_ax(roi_vel, roi_range, 'Power (dB)');
title('Range-Speed Response', 'FontWeight', 'bold', 'FontSize', 12);

% Panel 2: CFAR threshold surface ────────────────────────────────────────────
ax(2) = subplot(1, 3, 2);
imagesc(vel_axis, range_axis, thr_dB);
clim(clim_sig);
style_ax(roi_vel, roi_range, 'Power (dB)');
title('CA-CFAR Adaptive Threshold', 'FontWeight', 'bold', 'FontSize', 11);

% Panel 3: RD power + detections + ground truth ──────────────────────────────
ax(3) = subplot(1, 3, 3);
imagesc(vel_axis, range_axis, pow_dB);
clim(clim_sig);
style_ax(roi_vel, roi_range, 'Power (dB)');
title('Range-Speed Response Pattern', 'FontWeight', 'bold', 'FontSize', 12);
hold on;

% Ground-truth: large red open circles
h_gt = plot(truth_vel, truth_range, 'o', ...
    'MarkerSize', 20, 'LineWidth', 2, ...
    'Color', [0.85 0.15 0.10], 'MarkerFaceColor', 'none');
for j = 1 : K_truth
    text(truth_vel(j) + diff(roi_vel)*0.015, ...
         truth_range(j) + diff(roi_range)*0.015, ...
         sprintf('T%d', j), ...
         'Color', [0.85 0.15 0.10], 'FontWeight', 'bold', 'FontSize', 9);
end

% CFAR detections (sinc-interpolated): white squares
if K_det > 0
    h_det = plot(det_vel, det_range, 'ws', ...
        'MarkerSize', 11, 'LineWidth', 1.8, 'MarkerFaceColor', 'none');
    legend([h_gt h_det], ...
           {sprintf('Targets of interest (%d)', K_truth), ...
            sprintf('CFAR detections (%d)',     K_det)}, ...
           'Location', 'southeast', 'FontSize', 9, ...
           'TextColor', 'w', 'Color', [0.15 0.15 0.15], 'EdgeColor', 'w');
else
    legend(h_gt, sprintf('Targets of interest (%d)', K_truth), ...
           'Location', 'northeast', 'FontSize', 9, ...
           'TextColor', 'w', 'Color', [0.15 0.15 0.15], 'EdgeColor', 'w');
end
hold off;

linkaxes(ax, 'xy');
sgtitle(sprintf('CA-CFAR Detection on Range-Doppler Map   ($N_{\\rm chip}$=%d,  $N_{\\rm prbs}$=%d)', ...
        N_chip, N_block), ...
        'FontWeight', 'bold', 'FontSize', 13);

%% ════════════════════════════════════════════════════════════════════════════
%  FIGURE 2 — Estimation accuracy vs ground truth
%% ════════════════════════════════════════════════════════════════════════════
fig2 = figure('Name', 'Estimation vs Truth', ...
              'Position', [120 120 1400 500], 'Color', 'w');

% Panel 1: Range scatter ─────────────────────────────────────────────────────
subplot(1, 3, 1);
% Identity line spanning the full ROI
plot(roi_range, roi_range, 'k--', 'LineWidth', 1.2);
hold on;
for j = 1 : K_truth
    if ~isnan(assoc(j))
        plot(truth_range(j), det_range(assoc(j)), 'bo', ...
             'MarkerSize', 8, 'LineWidth', 1.5);
        text(truth_range(j), det_range(assoc(j)), ...
             sprintf('  T%d', j), 'FontSize', 8, 'Color', 'b');
    end
end
hold off;
xlabel('True range (m)',      'FontSize', 11);
ylabel('Estimated range (m)', 'FontSize', 11);
title('Range Estimation', 'FontWeight', 'bold', 'FontSize', 12);
legend({'Ideal ($\hat{R}=R$)', 'Detections'}, ...
       'Location', 'northwest', 'FontSize', 9);
grid on;  box on;  axis equal;
xlim(roi_range);  ylim(roi_range);

% Panel 2: Velocity scatter ──────────────────────────────────────────────────
subplot(1, 3, 2);
plot(roi_vel, roi_vel, 'k--', 'LineWidth', 1.2);
hold on;
for j = 1 : K_truth
    if ~isnan(assoc(j))
        plot(truth_vel(j), det_vel(assoc(j)), 'ro', ...
             'MarkerSize', 8, 'LineWidth', 1.5);
        text(truth_vel(j), det_vel(assoc(j)), ...
             sprintf('  T%d', j), 'FontSize', 8, 'Color', 'r');
    end
end
hold off;
xlabel('True velocity (m/s)',      'FontSize', 11);
ylabel('Estimated velocity (m/s)', 'FontSize', 11);
title('Velocity Estimation', 'FontWeight', 'bold', 'FontSize', 12);
legend({'Ideal ($\hat{v}=v$)', 'Detections'}, ...
       'Location', 'northwest', 'FontSize', 9);
grid on;  box on;  axis equal;
xlim(roi_vel);  ylim(roi_vel);

% Panel 3: Per-target error bar chart ────────────────────────────────────────
subplot(1, 3, 3);
bar_data = [err_R_vec, err_v_vec];   % (K_truth × 2), NaN = no association
b = bar(bar_data, 'grouped');
b(1).FaceColor = [0.20 0.45 0.80];
b(2).FaceColor = [0.85 0.33 0.10];
set(gca, 'XTick', 1:K_truth, ...
         'XTickLabel', arrayfun(@(j) sprintf('T%d',j), 1:K_truth, 'uni', 0));
xlabel('Target', 'FontSize', 11);
ylabel('Error', 'FontSize', 11);
title('Per-target Estimation Error', 'FontWeight', 'bold', 'FontSize', 12);
legend({'$\Delta r$ (m)', '$\Delta \nu$  (m/s)'}, 'Location', 'best', 'FontSize', 9);
yline(0, 'k--', 'LineWidth', 1);
grid on;  box on;
ylim(roi_vel);
sgtitle('Estimation Accuracy vs Ground Truth (Sinc Interpolation)', ...
        'FontWeight', 'bold', 'FontSize', 13);

%% ════════════════════════════════════════════════════════════════════════════
%  FIGURE 3 — 1-D slices at each detection
%% ════════════════════════════════════════════════════════════════════════════
if K_det > 0
    n_show = min(K_det, N_tars);

    fig3 = figure('Name', 'Per-detection slices', ...
                  'Position', [160 160 min(300*n_show, 1600) 620], 'Color', 'w');

    for k = 1 : n_show
        r_bin = detected_positions(k, 1);   % integer range   bin
        d_bin = detected_positions(k, 2);   % integer Doppler bin

        % Range slice at the detected Doppler bin ────────────────────────────
        slice_r      = pow_dB(:, d_bin);
        peak_slice_r = max(slice_r(range_mask));
        slice_r_norm = slice_r - peak_slice_r;   % normalise to 0 dB peak

        subplot(2, n_show, k);
        plot(range_axis, slice_r_norm, 'b', 'LineWidth', 1.2);
        hold on;
        xline(det_range(k), 'r--', 'LineWidth', 1.4);            % interpolated
        xline(range_axis(r_bin), 'g:', 'LineWidth', 1.0);        % integer bin
        hold off;
        xlabel('Range (m)',   'FontSize', 9);
        ylabel('Power (dB)',  'FontSize', 9);
        title(sprintf('Det\\,%d  $\\hat{v}$=%.1f\\,m/s', k, det_vel(k)), ...
              'FontSize', 9);
        legend({'Slice','Interp.','Bin'}, 'FontSize', 7, 'Location', 'best');
        xlim(roi_range);  ylim([-70 3]);
        grid on;  box on;

        % Velocity slice at the detected range bin ───────────────────────────
        slice_v      = pow_dB(r_bin, :);
        peak_slice_v = max(slice_v);
        slice_v_norm = slice_v - peak_slice_v;

        subplot(2, n_show, n_show + k);
        plot(vel_axis, slice_v_norm, 'b', 'LineWidth', 1.2);
        hold on;
        xline(det_vel(k),      'r--', 'LineWidth', 1.4);         % interpolated
        xline(vel_axis(d_bin), 'g:',  'LineWidth', 1.0);         % integer bin
        hold off;
        xlabel('Speed (m/s)', 'FontSize', 9);
        ylabel('Power (dB)',  'FontSize', 9);
        title(sprintf('Det\\,%d  $\\hat{R}$=%.1f\\,m', k, det_range(k)), ...
              'FontSize', 9);
        legend({'Slice','Interp.','Bin'}, 'FontSize', 7, 'Location', 'best');
        xlim(roi_vel);  ylim([-70 3]);
        grid on;  box on;
    end

    sgtitle('Range and Doppler slices at each detection (\color{red}red = interpolated, \color{green}green = integer bin)', ...
        'Interpreter', 'tex', ...
        'FontWeight', 'bold', ...
        'FontSize', 12);
end

%% ── Console summary ──────────────────────────────────────────────────────────
fprintf('\n══ CA-CFAR Detection Summary ══\n');
fprintf('  ROI  : Range [%.0f %.0f] m   Velocity [%.0f %.0f] m/s\n', ...
    roi_range(1), roi_range(2), roi_vel(1), roi_vel(2));
fprintf('  Guard [r=%d d=%d]  Train [r=%d d=%d]  Pfa=%.0e\n', ...
    N_guard_range, N_guard_doppler, N_train_range, N_train_doppler, P_fa);
fprintf('  Detections : %d\n', K_det);
if K_det > 0
    pwr_det   = RD_power_map(sub2ind(size(RD_power_map), ...
                    detected_positions(:,1), detected_positions(:,2)));
    noise_det = noise_power_avg_map(sub2ind(size(noise_power_avg_map), ...
                    detected_positions(:,1), detected_positions(:,2)));
    snr_det   = 10 * log10(pwr_det ./ (noise_det + 1e-20));

    fprintf('  %-5s  %-12s  %-14s  %-10s  %-12s  %-10s\n', ...
            'Rank','Range(m)','Speed(m/s)','Power(dB)','Noise(dB)','SNR(dB)');
    for k = 1 : K_det
        fprintf('%-5d  %-12.3f  %-14.3f  %-10.1f  %-12.1f  %-10.2f\n', ...
            k, det_range(k), det_vel(k), ...
            10*log10(pwr_det(k)   + eps), ...
            10*log10(noise_det(k) + eps), ...
            snr_det(k));
    end
end

%% ── Ground-truth SNR (using associated interpolated detection) ───────────────
fprintf('  Ground truth (%d targets):\n', K_truth);
fprintf('    %-4s  %-12s  %-14s  %-12s  %-12s  %-14s  %-12s  %-10s  %-12s  %-10s\n', ...
    'Tag', ...
    'R_true(m)', 'R_inter(m)', 'R_est(m)', ...
    'V_true(m/s)', 'V_inter(m/s)', 'V_est(m/s)', ...
    'Power(dB)', 'Noise(dB)', 'SNR(dB)');

for j = 1 : K_truth
    r0 = detected_positions(assoc(j), 1);
    d0 = detected_positions(assoc(j), 2);
    pwr_truth   = RD_power_map(r0, d0);
    noise_truth = noise_power_avg_map(r0, d0);
    snr_truth   = 10 * log10(pwr_truth / (noise_truth + 1e-20));

    r_est_inter = det_range(assoc(j));
    v_est_inter = det_vel(assoc(j));
    r_est       = range_axis(r0);
    v_est       = vel_axis(d0);

    fprintf('    T%-3d  %-12.4f  %-14.4f  %-12.4f  %-12.4f  %-14.4f  %-12.4f  %-10.1f  %-12.1f  %-10.2f\n', ...
        j, ...
        truth_range(j), r_est_inter, r_est, ...
        truth_vel(j),   v_est_inter, v_est, ...
        10*log10(pwr_truth   + eps), ...
        10*log10(noise_truth + eps), ...
        snr_truth);
end
fprintf('═══════════════════════════════\n\n');
% ---- Export figures (.fig + .png) into fig_exported ----
if ~exist('fig_dir','var')
    fig_dir = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
end
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
savefig(fig1, fullfile(fig_dir,'fig_ca_cfar_interpolate_rd_map.fig'));
exportgraphics(fig1, fullfile(fig_dir,'fig_ca_cfar_interpolate_rd_map.png'), 'Resolution',300);
savefig(fig2, fullfile(fig_dir,'fig_ca_cfar_est_vs_truth.fig'));
exportgraphics(fig2, fullfile(fig_dir,'fig_ca_cfar_est_vs_truth.png'), 'Resolution',300);
if exist('fig3','var') && isvalid(fig3)
    savefig(fig3, fullfile(fig_dir,'fig_ca_cfar_slices.fig'));
    exportgraphics(fig3, fullfile(fig_dir,'fig_ca_cfar_slices.png'), 'Resolution',300);
end

end

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function style_ax(roi_vel, roi_range, cbar_label)
    set(gca, 'YDir', 'normal');
    colormap(gca, parula);
    cb = colorbar;
    cb.Label.String   = cbar_label;
    cb.Label.FontSize = 11;
    xlim(roi_vel);
    ylim(roi_range);
    xlabel('Speed (m/s)',    'FontSize', 11);
    ylabel('Range (meters)', 'FontSize', 11);
    grid on;  box on;
    set(gca, 'GridColor', [0 0 0]*0.3, 'GridAlpha', 0.2, 'FontSize', 10);
end
