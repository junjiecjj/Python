% MAIN5B_SENSING_ANALYSIS  —  Waveform-parameter RMSE sweeps (companion to main5)
% -------------------------------------------------------------------------
% Reference: main5_sensing_analysis.m (same scene overrides, same estimator,
% same bootstrap-CI machinery). This driver ADDS TWO FIGURES:
%
%   Figure A   fig_rmse_range_vs_snr_nchip_sweep
%       Numerical range RMSE vs SNR for N_chip = [64 128 256 512] with
%       N_prbs (= N_block in the code) FIXED at the config value.
%
%   Figure B   fig_rmse_vel_vs_snr_nprbs_sweep
%       Numerical velocity RMSE vs SNR for N_prbs (= N_block) =
%       [64 128 256 512] with N_chip FIXED at the config value.
%
% Each sweep line is drawn in the style of fig_bias_significance_vs_snr in
% main5: a bootstrap 95% confidence band (lo, hi) as a shaded fill plus the
% empirical mid line, one (band + line) per N_chip / N_prbs value.
%
% "Numerical RMSE" = the CONVENTIONAL sensing pipeline of the paper:
% matched filter -> RD map (main4) -> CA-CFAR detection -> sinc
% interpolation -> nearest-truth association, exactly as in main5's CFAR
% branch. Residuals are pooled across the N_tars targets per SNR; missed
% detections give NaN residuals and are omitted from the bootstrap.
%
% Outputs (fig_exported / exported_statistics):
%   fig_rmse_range_vs_snr_nchip_sweep.(fig|png)
%   fig_rmse_vel_vs_snr_nprbs_sweep.(fig|png)
%   statistics_sensing_sweep_nchip.csv
%   statistics_sensing_sweep_nprbs.csv
%
% NOTE: long Monte-Carlo sweep (numel(sweep) x numel(SNR) x Mc full scene
% simulations per figure); run standalone.

clear; clc; close all;

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
stat_dir = fullfile(fileparts(mfilename('fullpath')), 'exported_statistics');
if ~exist(stat_dir,'dir'), mkdir(stat_dir); end

NO_SIGNAL_PLOT = true;
NO_RADAR_PLOT  = true;
NO_CFAR_PLOT   = true;

rng("default");

% =====================================================================
%  Scene + system setup (identical to main5_sensing_analysis)
% =====================================================================
TOPO_NO_PLOT = true; main2_topology;
UE_position = sensing_statistics_cfg.scene_override.ue_position_m(:);
UE_vel = sensing_statistics_cfg.scene_override.ue_velocity_mps(:);
UE_motion = phased.Platform('InitialPosition',UE_position,'Velocity',UE_vel);
Tars_vel(:,1:end-1) = sensing_statistics_cfg.scene_override.target_velocities_mps;
Tars_position(:,end) = UE_position;
Tars_vel(:,end) = UE_vel;
Tars_motion = phased.Platform('InitialPosition',Tars_position,'Velocity',Tars_vel);
G_tx_db = sensing_statistics_cfg.system_override.gain_tx_db;
G_rx_db = sensing_statistics_cfg.system_override.gain_rx_db;
G_ue_db = sensing_statistics_cfg.system_override.gain_ue_db;
N_F_com_db = sensing_statistics_cfg.system_override.noise_figure_com_db;
G_tx = 10^(G_tx_db/10); G_rx = 10^(G_rx_db/10); G_ue = 10^(G_ue_db/10);
N_F_com = 10^(N_F_com_db/10);
Noise_power_com = physconst('Boltzmann')*T_ref*N_F_com*B;
SNR_Tx_db = sensing_statistics_cfg.snr_db(:).';
SNR_Tx = 10.^(SNR_Tx_db / 10);
Mc = sensing_statistics_cfg.monte_carlo;

% Ground truth (fixed initial geometry, as in main5)
truth_range = sqrt(sum((Tars_position - Tx_position).^2, 1)).';
unit_vec    = (Tars_position - Tx_position) ./ ...
              (sqrt(sum((Tars_position - Tx_position).^2, 1)) + eps);
truth_vel   = sum(Tars_vel .* unit_vec, 1).';

% =====================================================================
%  Sweep settings
% =====================================================================
N_chip_fix   = N_chip;     % config value (fixed during Sweep B)
N_block_fix  = N_block;    % config value = N_prbs (fixed during Sweep A)

N_chip_sweep  = [64 128 256 512];   % Figure A (range RMSE)
N_prbs_sweep  = [64 128 256 512];   % Figure B (velocity RMSE), N_prbs = N_block

n_boot   = 2000;
alpha_ci = 0.05;           % 95% CI, as fig_bias_significance_vs_snr

n_snr = numel(SNR_Tx);

% =====================================================================
%  SWEEP A — range RMSE vs SNR for each N_chip (N_prbs fixed)
% =====================================================================
rmse_R_lo_A  = nan(n_snr, numel(N_chip_sweep));
rmse_R_hi_A  = nan(n_snr, numel(N_chip_sweep));
rmse_R_mid_A = nan(n_snr, numel(N_chip_sweep));

fprintf('===== Sweep A: N_chip in [%s], N_prbs = %d fixed =====\n', ...
    num2str(N_chip_sweep), N_block_fix);

for idx_cfg = 1:numel(N_chip_sweep)
    % ---- override the waveform length + its derived timing constants ----
    N_chip  = N_chip_sweep(idx_cfg);
    N_block = N_block_fix;
    if mod(N_chip, 2 * N_sym_per_block) ~= 0
        error("PDISAC:InvalidSweep", ...
            "N_chip = %d must be divisible by 2*N_sym_per_block = %d.", ...
            N_chip, 2 * N_sym_per_block);
    end
    T_pmcw         = N_chip * T_chip;
    L_slot_per_sym = N_chip / (2 * N_sym_per_block);
    N_slot         = N_chip / L_slot_per_sym;

    err_R_store = cell(n_snr, 1);   % pooled CFAR+sinc range residuals per SNR

    for idx_snr = 1:n_snr
        P_tx = Noise_power_sen * SNR_Tx(idx_snr);
        all_err_R_cfar = nan(Mc, N_tars);

        for idx_mc = 1:Mc
            main3_signal_channel_model;
            main4_sensing_process;

            % RD-map axes for the CURRENT (N_chip, N_block)
            range_res  = c / (2 * B);
            range_axis = (0 : N_chip-1) * range_res;
            PRF        = 1 / T_pmcw;
            fd_axis    = (-N_block/2 : N_block/2-1) * (PRF / N_block);
            vel_axis   = -(Lambda / 2) * fd_axis;

            % CA-CFAR detection + sinc interpolation (conventional pipeline)
            [~, ~, detected_positions, ~] = func_ca_cfar_adaptive_threshold( ...
                RD_map_shifted, ...
                N_guard_range, N_guard_doppler, ...
                N_train_range, N_train_doppler, ...
                P_fa, peak_select);

            err_R_vec = nan(N_tars, 1);
            K_det = size(detected_positions, 1);
            if K_det > 0
                [det_range, det_vel] = func_sinc_interpolation( ...
                    RD_map_shifted, detected_positions, range_axis, vel_axis);
                range_span = max(diff(Region_of_interest(1, :)), eps);
                vel_span   = max(diff(Region_of_interest(2, :)), eps);
                for j = 1:N_tars
                    dist2 = ((det_range - truth_range(j)) / range_span).^2 + ...
                            ((det_vel   - truth_vel(j)  ) / vel_span  ).^2;
                    [~, jj] = min(dist2);
                    err_R_vec(j) = det_range(jj) - truth_range(j);
                end
            end
            all_err_R_cfar(idx_mc, :) = err_R_vec;
        end

        err_R_store{idx_snr} = all_err_R_cfar;
        fprintf('  N_chip=%4d | SNR=%+3d dB | RMSE_R = %.4f m\n', ...
            N_chip, SNR_Tx_db(idx_snr), ...
            sqrt(mean(all_err_R_cfar(:).^2, 'omitnan')));
    end

    rng(1);  % reproducible bootstrap (same convention as main5)
    [lo, hi, mid] = local_bootstrap_rmse_ci(err_R_store, n_boot, alpha_ci);
    rmse_R_lo_A(:, idx_cfg)  = lo;
    rmse_R_hi_A(:, idx_cfg)  = hi;
    rmse_R_mid_A(:, idx_cfg) = mid;
end

% =====================================================================
%  SWEEP B — velocity RMSE vs SNR for each N_prbs (= N_block), N_chip fixed
% =====================================================================
rmse_v_lo_B  = nan(n_snr, numel(N_prbs_sweep));
rmse_v_hi_B  = nan(n_snr, numel(N_prbs_sweep));
rmse_v_mid_B = nan(n_snr, numel(N_prbs_sweep));

% restore the fixed chip length + its derived timing constants
N_chip         = N_chip_fix;
T_pmcw         = N_chip * T_chip;
L_slot_per_sym = N_chip / (2 * N_sym_per_block);
N_slot         = N_chip / L_slot_per_sym;

fprintf('===== Sweep B: N_prbs (N_block) in [%s], N_chip = %d fixed =====\n', ...
    num2str(N_prbs_sweep), N_chip_fix);

for idx_cfg = 1:numel(N_prbs_sweep)
    N_block = N_prbs_sweep(idx_cfg);

    err_v_store = cell(n_snr, 1);   % pooled CFAR+sinc velocity residuals per SNR

    for idx_snr = 1:n_snr
        P_tx = Noise_power_sen * SNR_Tx(idx_snr);
        all_err_v_cfar = nan(Mc, N_tars);

        for idx_mc = 1:Mc
            main3_signal_channel_model;
            main4_sensing_process;

            % RD-map axes for the CURRENT (N_chip, N_block)
            range_res  = c / (2 * B);
            range_axis = (0 : N_chip-1) * range_res;
            PRF        = 1 / T_pmcw;
            fd_axis    = (-N_block/2 : N_block/2-1) * (PRF / N_block);
            vel_axis   = -(Lambda / 2) * fd_axis;

            % CA-CFAR detection + sinc interpolation (conventional pipeline)
            [~, ~, detected_positions, ~] = func_ca_cfar_adaptive_threshold( ...
                RD_map_shifted, ...
                N_guard_range, N_guard_doppler, ...
                N_train_range, N_train_doppler, ...
                P_fa, peak_select);

            err_v_vec = nan(N_tars, 1);
            K_det = size(detected_positions, 1);
            if K_det > 0
                [det_range, det_vel] = func_sinc_interpolation( ...
                    RD_map_shifted, detected_positions, range_axis, vel_axis);
                range_span = max(diff(Region_of_interest(1, :)), eps);
                vel_span   = max(diff(Region_of_interest(2, :)), eps);
                for j = 1:N_tars
                    dist2 = ((det_range - truth_range(j)) / range_span).^2 + ...
                            ((det_vel   - truth_vel(j)  ) / vel_span  ).^2;
                    [~, jj] = min(dist2);
                    err_v_vec(j) = det_vel(jj) - truth_vel(j);
                end
            end
            all_err_v_cfar(idx_mc, :) = err_v_vec;
        end

        err_v_store{idx_snr} = all_err_v_cfar;
        fprintf('  N_prbs=%4d | SNR=%+3d dB | RMSE_v = %.4f m/s\n', ...
            N_block, SNR_Tx_db(idx_snr), ...
            sqrt(mean(all_err_v_cfar(:).^2, 'omitnan')));
    end

    rng(1);  % reproducible bootstrap (same convention as main5)
    [lo, hi, mid] = local_bootstrap_rmse_ci(err_v_store, n_boot, alpha_ci);
    rmse_v_lo_B(:, idx_cfg)  = lo;
    rmse_v_hi_B(:, idx_cfg)  = hi;
    rmse_v_mid_B(:, idx_cfg) = mid;
end

% restore the fixed block count for anything run after this script
N_block = N_block_fix;

% =====================================================================
%  Export sweep statistics (CSV)
% =====================================================================
tbl_A = local_sweep_table('N_chip', N_chip_sweep, SNR_Tx_db, ...
    rmse_R_lo_A, rmse_R_mid_A, rmse_R_hi_A, 'RMSE_Range_m');
writetable(tbl_A, fullfile(stat_dir, 'statistics_sensing_sweep_nchip.csv'));

tbl_B = local_sweep_table('N_prbs', N_prbs_sweep, SNR_Tx_db, ...
    rmse_v_lo_B, rmse_v_mid_B, rmse_v_hi_B, 'RMSE_Vel_mps');
writetable(tbl_B, fullfile(stat_dir, 'statistics_sensing_sweep_nprbs.csv'));

fprintf('\nSweep statistics saved to %s\n', stat_dir);

% =====================================================================
%  FIGURE A — Range RMSE vs SNR, one (lo, mid, hi) band per N_chip
%  Style: fig_bias_significance_vs_snr (95%% CI fill + mid line)
% =====================================================================
c_lines = [0.00 0.45 0.74;   % blue
           0.85 0.33 0.10;   % orange
           0.47 0.67 0.19;   % green
           0.49 0.18 0.56];  % purple
markers = {'o','s','^','d'};

x = SNR_Tx_db(:);

fig_nchip = figure('Name','Range RMSE vs SNR for different N_chip (95% CI)', ...
    'Position',[140 140 800 520],'Color','w');
hold on;
for idx_cfg = 1:numel(N_chip_sweep)
    ci = c_lines(mod(idx_cfg-1, size(c_lines,1)) + 1, :);
    lo = rmse_R_lo_A(:, idx_cfg);
    hi = rmse_R_hi_A(:, idx_cfg);
    vd = ~isnan(lo) & ~isnan(hi);
    fill([x(vd); flipud(x(vd))], [lo(vd); flipud(hi(vd))], ci, ...
        'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, rmse_R_mid_A(:, idx_cfg), ['-' markers{idx_cfg}], ...
        'LineWidth', 2.0, 'Color', ci, 'MarkerFaceColor', ci, ...
        'DisplayName', sprintf('$N_{\\rm chip} = %d$', N_chip_sweep(idx_cfg)));
end
set(gca, 'YScale', 'log');
grid on; box on;
xlabel('SNR (dB)');
ylabel('RMSE range (m)');
title(sprintf('Numerical Range RMSE vs SNR ($N_{\\rm prbs} = %d$ fixed, 95\\%% CI bands)', ...
    N_block_fix), 'Interpreter','latex');
legend('Location','best', 'Interpreter','latex');

savefig(fig_nchip, fullfile(fig_dir,'fig_rmse_range_vs_snr_nchip_sweep.fig'));
exportgraphics(fig_nchip, fullfile(fig_dir,'fig_rmse_range_vs_snr_nchip_sweep.png'),'Resolution',300);

% =====================================================================
%  FIGURE B — Velocity RMSE vs SNR, one (lo, mid, hi) band per N_prbs
% =====================================================================
fig_nprbs = figure('Name','Velocity RMSE vs SNR for different N_prbs (95% CI)', ...
    'Position',[160 160 800 520],'Color','w');
hold on;
for idx_cfg = 1:numel(N_prbs_sweep)
    ci = c_lines(mod(idx_cfg-1, size(c_lines,1)) + 1, :);
    lo = rmse_v_lo_B(:, idx_cfg);
    hi = rmse_v_hi_B(:, idx_cfg);
    vd = ~isnan(lo) & ~isnan(hi);
    fill([x(vd); flipud(x(vd))], [lo(vd); flipud(hi(vd))], ci, ...
        'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, rmse_v_mid_B(:, idx_cfg), ['-' markers{idx_cfg}], ...
        'LineWidth', 2.0, 'Color', ci, 'MarkerFaceColor', ci, ...
        'DisplayName', sprintf('$N_{\\rm prbs} = %d$', N_prbs_sweep(idx_cfg)));
end
set(gca, 'YScale', 'log');
grid on; box on;
xlabel('SNR (dB)');
ylabel('RMSE velocity (m/s)');
title(sprintf('Numerical Velocity RMSE vs SNR ($N_{\\rm chip} = %d$ fixed, 95\\%% CI bands)', ...
    N_chip_fix), 'Interpreter','latex');
legend('Location','best', 'Interpreter','latex');

savefig(fig_nprbs, fullfile(fig_dir,'fig_rmse_vel_vs_snr_nprbs_sweep.fig'));
exportgraphics(fig_nprbs, fullfile(fig_dir,'fig_rmse_vel_vs_snr_nprbs_sweep.png'),'Resolution',300);

fprintf('Figures saved to %s\n', fig_dir);

% =====================================================================
function [lo, hi, mid] = local_bootstrap_rmse_ci(err_store, n_boot, alpha_ci)
% Percentile bootstrap CI on the RMSE statistic, pooling trials across
% targets at each SNR point in err_store (a cell array, one Mc x N_tars
% error matrix per SNR). Identical to the helper in main5_sensing_analysis.
    n_snr = numel(err_store);
    lo  = nan(n_snr, 1);
    hi  = nan(n_snr, 1);
    mid = nan(n_snr, 1);
    for idx_snr = 1:n_snr
        err_vec = err_store{idx_snr}(:);
        err_vec = err_vec(~isnan(err_vec));
        n = numel(err_vec);
        if n == 0, continue; end

        boot_rmse = zeros(n_boot, 1);
        for b = 1:n_boot
            samp = err_vec(randi(n, n, 1));
            boot_rmse(b) = sqrt(mean(samp.^2));
        end
        boot_rmse_sorted = sort(boot_rmse);
        lo_idx = max(1, round(n_boot * (alpha_ci/2)));
        hi_idx = min(n_boot, round(n_boot * (1 - alpha_ci/2)));

        lo(idx_snr)  = boot_rmse_sorted(lo_idx);
        hi(idx_snr)  = boot_rmse_sorted(hi_idx);
        mid(idx_snr) = sqrt(mean(err_vec.^2));
    end
end

function tbl = local_sweep_table(param_name, param_vals, snr_db, lo, mid, hi, metric_name)
% Long-format table: one row per (sweep value, SNR) with lo/mid/hi RMSE.
    n_snr = numel(snr_db);
    n_cfg = numel(param_vals);
    rows  = n_snr * n_cfg;
    P  = zeros(rows,1); S = zeros(rows,1);
    L  = zeros(rows,1); M = zeros(rows,1); H = zeros(rows,1);
    r = 1;
    for idx_cfg = 1:n_cfg
        for idx_snr = 1:n_snr
            P(r) = param_vals(idx_cfg);
            S(r) = snr_db(idx_snr);
            L(r) = lo(idx_snr, idx_cfg);
            M(r) = mid(idx_snr, idx_cfg);
            H(r) = hi(idx_snr, idx_cfg);
            r = r + 1;
        end
    end
    tbl = table(P, S, L, M, H, 'VariableNames', ...
        {param_name, 'SNR_dB', [metric_name '_CI_lo'], ...
         [metric_name '_mid'], [metric_name '_CI_hi']});
end
