% MAIN9_ML_SENSING  —  Conventional vs ML-based (RDPDNet) sensing comparison
% -------------------------------------------------------------------------
% Runs a Monte-Carlo SNR sweep in which every scene is processed by BOTH
% sensing pipelines and scored against the same ground truth and CRLB.
%
% *** ENHANCEMENT over main9_ml_sensing ***
% This script incorporates the estimator bias into the CRLB comparison.
% Since MSE = Var + bias^2, and an estimator's variance is bounded by the
% unbiased CRLB (approximately, assuming small bias gradient), we compare 
% the empirical RMSE = sqrt(MSE) against sqrt(CRLB_unbiased + empirical_bias^2).
% This provides a mathematically fairer bound for biased estimators (like RDPDNet).
%
%   received signal --> matched filter --> complex RD map        (shared)
%        |                                                        main3/main4
%        |----> [conventional] CA-CFAR --> sinc interp --> (r,v)  MATLAB
%        |
%        '----> [ML] RDPDNet denoise --> CA-CFAR --> sinc interp --> (r,v)
%                     (trained checkpoint, Python bridge)
%
% Outputs (in fig_exported / exported_statistics):
%   fig_ml_vs_conv_rmse_vs_snr_biased_crlb.(fig|png)   range/vel RMSE vs SNR
%   fig_ml_vs_conv_pd_vs_snr_biased_crlb.(fig|png)     detection prob vs SNR
%   fig_ml_vs_conv_heatmap_cfar_biased_crlb.(fig|png)  RD/CFAR comparison
%   statistics_ml_vs_conv_sensing_biased_crlb.csv      per-SNR summary table

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultColorbarTickLabelInterpreter','latex');
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
stat_dir = fullfile(fileparts(mfilename('fullpath')), 'exported_statistics');
if ~exist(stat_dir,'dir'), mkdir(stat_dir); end

% Run every sub-script headless (they are called once per scene).
NO_SIGNAL_PLOT = true; NO_RADAR_PLOT = true; NO_CFAR_PLOT = true;

% =====================================================================
%  1) Configuration and fixed scene
% =====================================================================
main1_matlab_config;

rng("default");
TOPO_NO_PLOT = true; main2_topology;

UE_position = sensing_statistics_cfg.scene_override.ue_position_m(:);
UE_vel      = sensing_statistics_cfg.scene_override.ue_velocity_mps(:);
UE_motion   = phased.Platform('InitialPosition',UE_position,'Velocity',UE_vel);
Tars_vel(:,1:end-1) = sensing_statistics_cfg.scene_override.target_velocities_mps;
Tars_position(:,end) = UE_position;
Tars_vel(:,end)      = UE_vel;
Tars_motion = phased.Platform('InitialPosition',Tars_position,'Velocity',Tars_vel);

G_tx_db = sensing_statistics_cfg.system_override.gain_tx_db;
G_rx_db = sensing_statistics_cfg.system_override.gain_rx_db;
G_ue_db = sensing_statistics_cfg.system_override.gain_ue_db;
G_tx = 10^(G_tx_db/10); G_rx = 10^(G_rx_db/10); G_ue = 10^(G_ue_db/10);

SNR_Tx_db = sensing_statistics_cfg.snr_db(:).';
SNR_Tx    = 10.^(SNR_Tx_db / 10);
Mc        = sensing_statistics_cfg.monte_carlo;
n_snr     = numel(SNR_Tx);

% =====================================================================
%  2) Pre-flight: RDPDNet (Python) backend must be available
% =====================================================================
python_exe_ml = string(PDISAC_cfg.inference.python_executable);
if ~(startsWith(python_exe_ml, filesep) || ...
     ~isempty(regexp(python_exe_ml, '^[A-Za-z]:[\\/]', 'once')) || ...
     isfile(python_exe_ml))
    python_exe_ml = fullfile(PDISAC_repo_root, python_exe_ml);
end
checkpoint_ml = fullfile(PDISAC_repo_root, string(PDISAC_cfg.inference.checkpoint));
if ~isfile(python_exe_ml) || ~isfile(checkpoint_ml)
    error('main9_ml_sensing:MLBackendMissing', ...
        ['RDPDNet backend not available; cannot run the ML branch.\n' ...
         '  python executable : %s  [%s]\n' ...
         '  RDPDNet checkpoint  : %s  [%s]\n' ...
         'Train RDPDNet (alg_pdisac) and/or set PDISAC_cfg.inference.* in ' ...
         'main1_matlab_config, then rerun main9_ml_sensing.'], ...
         python_exe_ml, tf(isfile(python_exe_ml)), ...
         checkpoint_ml, tf(isfile(checkpoint_ml)));
end

% =====================================================================
%  3) Shared axes, CFAR settings, ground truth, and detection gate
% =====================================================================
range_res  = c / (2 * B);
range_axis = (0:N_chip-1) * range_res;
PRF        = 1 / T_pmcw;
fd_axis    = (-N_block/2 : N_block/2-1) * (PRF / N_block);
vel_axis   = -(Lambda / 2) * fd_axis;
vel_res    = (Lambda / 2) * (PRF / N_block);

N_g_r = PDISAC_cfg.sensing.cfar.guard_range;
N_g_d = PDISAC_cfg.sensing.cfar.guard_doppler;
N_t_r = PDISAC_cfg.sensing.cfar.training_range;
N_t_d = PDISAC_cfg.sensing.cfar.training_doppler;
P_fa  = PDISAC_cfg.sensing.cfar.false_alarm_probability;
peak_select = PDISAC_cfg.sensing.cfar.peak_select;

truth_range = sqrt(sum((Tars_position - Tx_position).^2, 1)).';
unit_vec    = (Tars_position - Tx_position) ./ ...
              (sqrt(sum((Tars_position - Tx_position).^2, 1)) + eps);
truth_vel   = sum(Tars_vel .* unit_vec, 1).';
N_tars      = numel(truth_range);

roi_range = Region_of_interest(1, :);
roi_vel   = Region_of_interest(2, :);

gate_r = 5 * range_res;
gate_v = 5 * vel_res;

% =====================================================================
%  4) Pass 1: build RD maps, score CONVENTIONAL branch, stash maps
% =====================================================================
n_scene = n_snr * Mc;
RD_stack     = complex(zeros(N_chip, N_block, n_scene, 'single'));
scene_snr    = zeros(n_scene, 1);
errR_conv    = nan(n_scene, N_tars);
errV_conv    = nan(n_scene, N_tars);
hit_conv     = false(n_scene, N_tars);
false_conv   = zeros(n_scene, 1);

fprintf('Pass 1/2: %d scenes (%d SNR x %d MC) — MF + conventional CFAR\n', ...
        n_scene, n_snr, Mc);
scene = 0;
for idx_snr = 1:n_snr
    P_tx = Noise_power_sen * SNR_Tx(idx_snr);
    for idx_mc = 1:Mc
        main3_signal_channel_model;
        main4_sensing_process;

        scene = scene + 1;
        RD_stack(:,:,scene) = RD_map_shifted;
        scene_snr(scene)    = idx_snr;

        [eR, eV, hit, n_false] = score_branch(RD_map_shifted, N_g_r, N_g_d, N_t_r, N_t_d, ...
            P_fa, peak_select, range_axis, vel_axis, ...
            truth_range, truth_vel, roi_range, roi_vel, gate_r, gate_v);
        errR_conv(scene,:) = eR;
        errV_conv(scene,:) = eV;
        hit_conv(scene,:)  = hit;
        false_conv(scene)  = n_false;
    end
    fprintf('   SNR = %+3d dB done\n', SNR_Tx_db(idx_snr));
end

% =====================================================================
%  5) Batched RDPDNet inference
% =====================================================================
fprintf('RDPDNet inference on %d RD maps (batched)...\n', n_scene);
RD_batch = permute(RD_stack, [3 1 2]);
RD_den_batch = inference_run_pdnet( ...
    RD_batch, string(checkpoint_ml), ...
    PythonExecutable = python_exe_ml, ...
    Device    = string(PDISAC_cfg.inference.device), ...
    BatchSize = PDISAC_cfg.inference.batch_size);

% =====================================================================
%  6) Pass 2: score ML branch on denoised RD maps
% =====================================================================
fprintf('Pass 2/2: RDPDNet-denoised CFAR scoring\n');
errR_ml = nan(n_scene, N_tars);
errV_ml = nan(n_scene, N_tars);
hit_ml  = false(n_scene, N_tars);
false_ml = zeros(n_scene, 1);
for s = 1:n_scene
    RD_den = squeeze(RD_den_batch(s,:,:));
    [eR, eV, hit, n_false] = score_branch(RD_den, N_g_r, N_g_d, N_t_r, N_t_d, ...
        P_fa, peak_select, range_axis, vel_axis, ...
        truth_range, truth_vel, roi_range, roi_vel, gate_r, gate_v);
    errR_ml(s,:) = eR;
    errV_ml(s,:) = eV;
    hit_ml(s,:)  = hit;
    false_ml(s)  = n_false;
end

% =====================================================================
%  7) Aggregate metrics over SNR (incorporating BIAS into CRLB)
% =====================================================================
rmseR_conv = nan(n_snr,1); rmseV_conv = nan(n_snr,1); pd_conv = nan(n_snr,1);
rmseR_ml   = nan(n_snr,1); rmseV_ml   = nan(n_snr,1); pd_ml   = nan(n_snr,1);
falsePerMap_conv = nan(n_snr,1); falsePerMap_ml = nan(n_snr,1);

% Adjusted CRLBs
crlbR_adj_conv = nan(n_snr,1); crlbV_adj_conv = nan(n_snr,1);
crlbR_adj_ml   = nan(n_snr,1); crlbV_adj_ml   = nan(n_snr,1);

for idx_snr = 1:n_snr
    rows = (scene_snr == idx_snr);

    rmseR_conv(idx_snr) = rmse_over(errR_conv(rows,:));
    rmseV_conv(idx_snr) = rmse_over(errV_conv(rows,:));
    pd_conv(idx_snr)    = mean(hit_conv(rows,:), 'all');
    falsePerMap_conv(idx_snr) = mean(false_conv(rows));

    rmseR_ml(idx_snr)   = rmse_over(errR_ml(rows,:));
    rmseV_ml(idx_snr)   = rmse_over(errV_ml(rows,:));
    pd_ml(idx_snr)      = mean(hit_ml(rows,:), 'all');
    falsePerMap_ml(idx_snr) = mean(false_ml(rows));

    % Get the unbiased CRLB variance for each target
    P_tx = Noise_power_sen * SNR_Tx(idx_snr);
    [cR, cV] = func_ana_crlb_rv( ...
        N_tars, Tars_position, Tars_rcs, Tars_vel, ...
        Tx_position, N_tx_ant, Rx_position, N_rx_ant, ...
        P_tx, G_tx, G_rx, Lambda, fc, Noise_power_sen, ...
        T_chip, N_chip, N_block);

    % Compute biased CRLB per target: sqrt2(CRLB + bias^2)
    cR_adj_conv_tars = zeros(N_tars, 1);
    cV_adj_conv_tars = zeros(N_tars, 1);
    cR_adj_ml_tars   = zeros(N_tars, 1);
    cV_adj_ml_tars   = zeros(N_tars, 1);

    for j = 1:N_tars
        % Empirical bias per target
        bR_conv = mean(errR_conv(rows, j), 'omitnan');
        bV_conv = mean(errV_conv(rows, j), 'omitnan');
        bR_ml   = mean(errR_ml(rows, j), 'omitnan');
        bV_ml   = mean(errV_ml(rows, j), 'omitnan');
        
        cR_adj_conv_tars(j) = cR(j) + bR_conv^2;
        cV_adj_conv_tars(j) = cV(j) + bV_conv^2;
        
        cR_adj_ml_tars(j)   = cR(j) + bR_ml^2;
        cV_adj_ml_tars(j)   = cV(j) + bV_ml^2;
    end

    % Average the adjusted RMSE bound across targets
    crlbR_adj_conv(idx_snr) = mean(sqrt(cR_adj_conv_tars), 'omitnan');
    crlbV_adj_conv(idx_snr) = mean(sqrt(cV_adj_conv_tars), 'omitnan');
    crlbR_adj_ml(idx_snr)   = mean(sqrt(cR_adj_ml_tars), 'omitnan');
    crlbV_adj_ml(idx_snr)   = mean(sqrt(cV_adj_ml_tars), 'omitnan');

    fprintf(['SNR=%+3d dB | RMSE_R conv=%.3f (adjCRLB=%.3f) ml=%.3f (adjCRLB=%.3f) | ' ...
             'RMSE_v conv=%.3f (adjCRLB=%.3f) ml=%.3f (adjCRLB=%.3f) | Pd ml=%.2f\n'], ...
        SNR_Tx_db(idx_snr), rmseR_conv(idx_snr), crlbR_adj_conv(idx_snr), ...
        rmseR_ml(idx_snr), crlbR_adj_ml(idx_snr), ...
        rmseV_conv(idx_snr), crlbV_adj_conv(idx_snr), ...
        rmseV_ml(idx_snr), crlbV_adj_ml(idx_snr), pd_ml(idx_snr));
end

% =====================================================================
%  8) Save summary table
% =====================================================================
summary = table(SNR_Tx_db(:), rmseR_conv, rmseR_ml, ...
                crlbR_adj_conv, crlbR_adj_ml, ...
                rmseV_conv, rmseV_ml, ...
                crlbV_adj_conv, crlbV_adj_ml, ...
                pd_conv, pd_ml, falsePerMap_conv, falsePerMap_ml, ...
    'VariableNames', {'SNR_dB', ...
        'RMSE_R_conv_m','RMSE_R_ml_m', ...
        'sqrtCRLB_adj_R_conv_m', 'sqrtCRLB_adj_R_ml_m', ...
        'RMSE_v_conv_mps','RMSE_v_ml_mps', ...
        'sqrtCRLB_adj_v_conv_mps', 'sqrtCRLB_adj_v_ml_mps', ...
        'Pd_conv','Pd_ml','FalseDetectionsPerMap_conv','FalseDetectionsPerMap_ml'});
csv_path = fullfile(stat_dir, 'statistics_ml_vs_conv_sensing_biased_crlb.csv');
writetable(summary, csv_path);
fprintf('\nSummary saved to %s\n', csv_path);

% =====================================================================
%  9) Plots: RMSE vs SNR (conv / ML / adjusted CRLB)
% =====================================================================
c_conv = [0.85 0.33 0.10];   % conventional
c_ml   = [0.00 0.45 0.74];   % RDPDNet (ML)

fig_rmse = figure('Name','ML vs Conventional: RMSE vs SNR (Biased CRLB)', ...
    'Position',[100 100 1400 520],'Color','w');

subplot(1,2,1);
semilogy(SNR_Tx_db, rmseR_conv, '--s','LineWidth',1.6,'Color',c_conv, 'MarkerFaceColor',c_conv); hold on;
semilogy(SNR_Tx_db, crlbR_adj_conv, ':s','LineWidth',1.2,'Color',c_conv.*0.7);
semilogy(SNR_Tx_db, rmseR_ml,   '-o','LineWidth',2.0,'Color',c_ml, 'MarkerFaceColor',c_ml);
semilogy(SNR_Tx_db, crlbR_adj_ml,   ':o','LineWidth',1.2,'Color',c_ml.*0.7);
grid on; box on;
xlabel('SNR (dB)'); ylabel('Conditional RMSE range (m)');
title('Conditional range RMSE vs SNR');
legend({'Conventional (CFAR)','$\sqrt{\mathrm{CRLB} + b_{conv}^2}$', ...
        'RDPDNet (ML)','$\sqrt{\mathrm{CRLB} + b_{ml}^2}$'}, 'Location','best');

subplot(1,2,2);
semilogy(SNR_Tx_db, rmseV_conv, '--s','LineWidth',1.6,'Color',c_conv, 'MarkerFaceColor',c_conv); hold on;
semilogy(SNR_Tx_db, crlbV_adj_conv, ':s','LineWidth',1.2,'Color',c_conv.*0.7);
semilogy(SNR_Tx_db, rmseV_ml,   '-o','LineWidth',2.0,'Color',c_ml, 'MarkerFaceColor',c_ml);
semilogy(SNR_Tx_db, crlbV_adj_ml,   ':o','LineWidth',1.2,'Color',c_ml.*0.7);
grid on; box on;
xlabel('SNR (dB)'); ylabel('Conditional RMSE velocity (m/s)');
title('Conditional velocity RMSE vs SNR');
legend({'Conventional (CFAR)','$\sqrt{\mathrm{CRLB} + b_{conv}^2}$', ...
        'RDPDNet (ML)','$\sqrt{\mathrm{CRLB} + b_{ml}^2}$'}, 'Location','best');

sgtitle('Sensing accuracy: conventional vs RDPDNet (Bias-Adjusted CRLB)');
savefig(fig_rmse, fullfile(fig_dir,'fig_ml_vs_conv_rmse_vs_snr_biased_crlb.fig'));
exportgraphics(fig_rmse, fullfile(fig_dir,'fig_ml_vs_conv_rmse_vs_snr_biased_crlb.png'),'Resolution',300);

fig_pd = figure('Name','ML vs Conventional: P_d vs SNR', ...
    'Position',[120 120 720 520],'Color','w');
plot(SNR_Tx_db, pd_conv, '--s','LineWidth',1.6,'Color',c_conv, 'MarkerFaceColor',c_conv); hold on;
plot(SNR_Tx_db, pd_ml,   '-o','LineWidth',2.0,'Color',c_ml, 'MarkerFaceColor',c_ml);
grid on; box on; ylim([-0.02 1.02]);
xlabel('SNR (dB)'); ylabel('Detection probability $P_d$');
title('Detection probability vs SNR');
legend({'Conventional (CFAR)','RDPDNet (ML)'}, 'Location','southeast');
savefig(fig_pd, fullfile(fig_dir,'fig_ml_vs_conv_pd_vs_snr_biased_crlb.fig'));
exportgraphics(fig_pd, fullfile(fig_dir,'fig_ml_vs_conv_pd_vs_snr_biased_crlb.png'),'Resolution',300);

% =====================================================================
% 10) Representative RD heatmaps and CA-CFAR response
% =====================================================================
[~, diagnostic_snr_idx] = max(SNR_Tx_db);
diagnostic_scene = find(scene_snr == diagnostic_snr_idx, 1, 'first');
RD_conv_example = double(RD_stack(:,:,diagnostic_scene));
RD_ml_example   = double(squeeze(RD_den_batch(diagnostic_scene,:,:)));

[thr_conv,~,pos_conv,pow_conv] = func_ca_cfar_adaptive_threshold( ...
    RD_conv_example, N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select);
[thr_ml,~,pos_ml,pow_ml] = func_ca_cfar_adaptive_threshold( ...
    RD_ml_example, N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select);

pow_conv_db = 10*log10(pow_conv + eps);
pow_ml_db   = 10*log10(pow_ml + eps);
margin_conv_db = 10*log10((pow_conv + eps) ./ (thr_conv + eps));
margin_ml_db   = 10*log10((pow_ml + eps) ./ (thr_ml + eps));
power_limits = [min([pow_conv_db(:); pow_ml_db(:)]), ...
                max([pow_conv_db(:); pow_ml_db(:)])];
margin_abs = max(abs([margin_conv_db(:); margin_ml_db(:)]));
margin_limits = [-margin_abs margin_abs];

fig_heatmap = figure('Name','Conventional vs RDPDNet: RD maps and CFAR', ...
    'Position',[80 60 1450 900],'Color','w');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot_rd_cfar_map(pow_conv_db, pos_conv, range_axis, vel_axis, ...
    truth_range, truth_vel, roi_range, roi_vel, power_limits, ...
    'Conventional noisy RD map');

nexttile;
plot_rd_cfar_map(pow_ml_db, pos_ml, range_axis, vel_axis, ...
    truth_range, truth_vel, roi_range, roi_vel, power_limits, ...
    'RDPDNet-predicted RD map');

nexttile;
plot_cfar_margin(margin_conv_db, pos_conv, range_axis, vel_axis, ...
    truth_range, truth_vel, roi_range, roi_vel, margin_limits, ...
    'Conventional: CA-CFAR margin');

nexttile;
plot_cfar_margin(margin_ml_db, pos_ml, range_axis, vel_axis, ...
    truth_range, truth_vel, roi_range, roi_vel, margin_limits, ...
    'RDPDNet: CA-CFAR margin');

sgtitle(sprintf('RD-map and identical CA-CFAR comparison at SNR = %+g dB', ...
    SNR_Tx_db(diagnostic_snr_idx)));
savefig(fig_heatmap, fullfile(fig_dir,'fig_ml_vs_conv_heatmap_cfar_biased_crlb.fig'));
exportgraphics(fig_heatmap, fullfile(fig_dir,'fig_ml_vs_conv_heatmap_cfar_biased_crlb.png'), ...
    'Resolution',300);

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function s = tf(b)
    if b, s = 'found'; else, s = 'MISSING'; end
end

function r = rmse_over(err_block)
    v = err_block(~isnan(err_block));
    if isempty(v), r = NaN; else, r = sqrt(mean(v.^2)); end
end

function plot_rd_cfar_map(power_db, det_pos, range_axis, vel_axis, ...
        truth_range, truth_vel, roi_range, roi_vel, color_limits, plot_title)
    imagesc(vel_axis, range_axis, power_db);
    axis xy; hold on; colormap(gca, parula); clim(color_limits);
    if ~isempty(det_pos)
        plot(vel_axis(det_pos(:,2)), range_axis(det_pos(:,1)), 'wo', ...
            'MarkerSize',7,'LineWidth',1.4,'DisplayName','CA-CFAR');
    end
    plot(truth_vel, truth_range, 'rx','MarkerSize',10,'LineWidth',2, ...
        'DisplayName','Truth');
    xlim(sort(roi_vel)); ylim(sort(roi_range)); grid on; box on;
    xlabel('Radial velocity (m/s)'); ylabel('Range (m)'); title(plot_title);
    cb = colorbar; cb.Label.String = 'RD power (dB)';
    legend('Location','best');
end

function plot_cfar_margin(margin_db, det_pos, range_axis, vel_axis, ...
        truth_range, truth_vel, roi_range, roi_vel, color_limits, plot_title)
    imagesc(vel_axis, range_axis, margin_db);
    axis xy; hold on; colormap(gca, turbo); clim(color_limits);
    if ~isempty(det_pos)
        plot(vel_axis(det_pos(:,2)), range_axis(det_pos(:,1)), 'ko', ...
            'MarkerFaceColor','w','MarkerSize',6,'LineWidth',1.1, ...
            'DisplayName','Passed CFAR');
    end
    plot(truth_vel, truth_range, 'rx','MarkerSize',10,'LineWidth',2, ...
        'DisplayName','Truth');
    xlim(sort(roi_vel)); ylim(sort(roi_range)); grid on; box on;
    xlabel('Radial velocity (m/s)'); ylabel('Range (m)'); title(plot_title);
    cb = colorbar; cb.Label.String = 'CFAR margin, 10log_{10}(P/T) (dB)';
    legend('Location','best');
end

function [errR, errV, hit, n_false] = score_branch(RD_map, N_g_r, N_g_d, N_t_r, N_t_d, ...
        P_fa, peak_select, range_axis, vel_axis, ...
        truth_range, truth_vel, roi_range, roi_vel, gate_r, gate_v)
    N_tars = numel(truth_range);
    errR = nan(N_tars,1); errV = nan(N_tars,1); hit = false(N_tars,1);
    n_false = 0;

    [~,~,det_pos,~] = func_ca_cfar_adaptive_threshold( ...
        RD_map, N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select);
    if isempty(det_pos), return; end

    [det_range, det_vel] = func_sinc_interpolation( ...
        RD_map, det_pos, range_axis, vel_axis);

    in_roi = det_range >= min(roi_range) & det_range <= max(roi_range) & ...
             det_vel   >= min(roi_vel)   & det_vel   <= max(roi_vel);
    det_range = det_range(in_roi);
    det_vel   = det_vel(in_roi);
    n_det = numel(det_range);
    if n_det == 0, return; end

    pairs = zeros(0,3);
    for j = 1:N_tars
        dR = det_range - truth_range(j);
        dV = det_vel   - truth_vel(j);
        candidate = find(abs(dR) <= gate_r & abs(dV) <= gate_v);
        cost = (dR(candidate)/gate_r).^2 + (dV(candidate)/gate_v).^2;
        pairs = [pairs; [repmat(j,numel(candidate),1), candidate(:), cost(:)]]; %#ok<AGROW>
    end
    if isempty(pairs)
        n_false = n_det;
        return;
    end

    [~, order] = sort(pairs(:,3), 'ascend');
    used_tar = false(N_tars,1);
    used_det = false(n_det,1);
    for k = order.'
        j = pairs(k,1);
        a = pairs(k,2);
        if used_tar(j) || used_det(a), continue; end
        used_tar(j) = true;
        used_det(a) = true;
        hit(j)  = true;
        errR(j) = det_range(a) - truth_range(j);
        errV(j) = det_vel(a)   - truth_vel(j);
    end
    n_false = n_det - nnz(used_det);
end
