% MAIN9B_ML_SENSING  —  Data-embedded vs data-free RD maps through RDPDNet
% -------------------------------------------------------------------------
% Uses a nearest-truth association per target (main5-style) with an acceptance
% gate: the nearest detection is assigned to a
% target only if |dR| <= 5*range_res AND |dV| <= 5*vel_res. A target whose
% nearest detection falls outside the gate (or with no detections at all,
% e.g. at very low SNR) is a MISS: its residual is NaN and does NOT count in
% the RMSE, which is therefore conditional on detection. Per-target hits are
% tracked, giving the detection probability P_d of each of the four
% branches. Bias-adjusted CRLB sqrt(CRLB + bias^2) uses the unbiased CRLB
% from func_ana_crlb_rv, per target, as in main5/main9b. Python side:
% alg_pdisac (RDPDNet + AFM MaskNet).
%
% This driver quantifies how much the embedded communication symbols degrade
% sensing, before and after RDPDNet denoising. Every Monte-Carlo scene yields
% TWO matched-filter RD maps (main4_sensing_process):
%
%   RD_map_shifted               MF with the DATA-carrying waveform ("data")
%   RD_map_noise_no_data_shifted MF with the pure PRBS reference  ("no data")
%
% and each is pushed through the two back-ends:
%
%   [conv] RD map ------------------> CA-CFAR -> sinc interp -> (r, v)
%   [ML]   RD map --> RDPDNet denoise -> CA-CFAR -> sinc interp -> (r, v)
%
% FIGURE 1a  fig_rd_maps   (1 x 3, highest SNR, representative scene)
%     (a) RD map with data symbols          |RD_map_shifted|
%     (b) RD map without data symbols       |RD_map_noise_no_data_shifted|
%     (c) RDPDNet-denoised RD map (input (a))
%
% FIGURE 1b  fig_afm_masks (1 x 3, same style as fig_rd_maps:
%            parula, dB magnitude, axes + colorbar + ROI limits + truth marks)
%     (a) M_mask = MaskNet(Z_rd_hat)
%     (b) M_tars = create_mask(Z_rd_prbs)
%     (c) M_afm  = M_mask .* (1 - M_tars)
%   Masks rebuilt from the trained checkpoint via inference_run_pdnet_mask.
%
% FIGURE 2  fig_rmse_range_vs_snr   (range)
% FIGURE 3  fig_rmse_vel_vs_snr     (velocity)
%   Mean RMSE vs SNR with FIVE lines each:
%     1. RDPDNet(data)    -> CFAR -> sinc
%     2. RDPDNet(no data) -> CFAR -> sinc
%     3. data           -> CFAR -> sinc          (conventional)
%     4. no data        -> CFAR -> sinc          (conventional)
%     5. sqrt(CRLB_unbiased + bias^2): unbiased (theoretical) CRLB from
%        func_ana_crlb_rv, bias from the conventional NO-DATA branch, per
%        target, then averaged over targets — exactly main5's adjusted bound.
%   All lines are the mean over the N_tars targets of per-target RMSE
%   (main5 "Mean RMSE vs SNR" convention).
%
% Outputs (fig_exported / exported_statistics):
%   fig_rd_maps.(fig|png)
%   fig_afm_masks.(fig|png)
%   fig_rmse_range_vs_snr.(fig|png)
%   fig_rmse_vel_vs_snr.(fig|png)
%   fig_pd_vs_snr.(fig|png)         detection probability, 4 branches
%   fig_bias_significance_vs_snr.(fig|png)  bias + 95% CI (conv no-data)
%   statistics_main9b_data_vs_nodata.csv
%
% Requirements: trained RDPDNet checkpoint WITH afm_model_state_dict and the
% Python env configured in main1_matlab_config (PDISAC_cfg.inference.*).

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultColorbarTickLabelInterpreter','latex');
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultAxesFontSize',13);
set(groot,'defaultLineLineWidth',1.6);

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
stat_dir = fullfile(fileparts(mfilename('fullpath')), 'exported_statistics');
if ~exist(stat_dir,'dir'), mkdir(stat_dir); end

NO_SIGNAL_PLOT = true; NO_RADAR_PLOT = true; NO_CFAR_PLOT = true;

% =====================================================================
%  1) Configuration and fixed scene  (identical to main9b)
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
    error('main9b_ml_sensing:MLBackendMissing', ...
        ['RDPDNet backend not available; cannot run the ML branches.\n' ...
         '  python executable : %s  [%s]\n' ...
         '  RDPDNet checkpoint  : %s  [%s]\n' ...
         'Train RDPDNet (alg_pdisac) and/or set PDISAC_cfg.inference.* in ' ...
         'main1_matlab_config, then rerun main9b_ml_sensing.'], ...
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

% ---- score_branch options ------------------------------------------------
% The gate and the ROI filter are both OPTIONAL:
%   USE_GATE = true  : the nearest detection is accepted for a target only if
%                      |dR| <= gate_r and |dV| <= gate_v. A target with no
%                      detection inside the gate (e.g. it does not appear on
%                      the RD map at low SNR) is a MISS: its residual is NaN
%                      and is NOT counted in the RMSE/bias.
%   USE_GATE = false : no distance cutoff, but the one-to-one assignment
%                      still applies; targets beyond the number of available
%                      detections remain NaN (miss).
%   USE_ROI  = true  : detections outside roi_range/roi_vel are discarded
%                      before association.
USE_GATE = false;
USE_ROI  = false;
gate_r = 20 * range_res;
gate_v = 20 * vel_res;

% =====================================================================
%  4) Pass 1: build BOTH RD maps per scene, score the CONV branches
% =====================================================================
n_scene = n_snr * Mc;
RD_stack_data   = complex(zeros(N_chip, N_block, n_scene, 'single'));
RD_stack_nodata = complex(zeros(N_chip, N_block, n_scene, 'single'));
scene_snr = zeros(n_scene, 1);

errR_conv_data   = nan(n_scene, N_tars);  errV_conv_data   = nan(n_scene, N_tars);
errR_conv_nodata = nan(n_scene, N_tars);  errV_conv_nodata = nan(n_scene, N_tars);
hit_conv_data    = false(n_scene, N_tars);
hit_conv_nodata  = false(n_scene, N_tars);

fprintf('Pass 1/2: %d scenes (%d SNR x %d MC) — MF + conventional CFAR (data & no-data)\n', ...
        n_scene, n_snr, Mc);
% Clean RD map (no noise, no data = Z_rd_prbs in training) captured for the
% diagnostic scene (first MC trial at the highest SNR) — used for M_tars.
[~, diag_snr_idx_pass1] = max(SNR_Tx_db);
RD_ex_clean = [];
% Clean maps for ALL trials at the diagnostic SNR: used as Z_rd^prbs by the
% RDPDNet distribution export (Section 13). Only one SNR is stored to bound memory.
RD_stack_clean_diag = complex(zeros(N_chip, N_block, Mc, 'single'));
scene = 0;
for idx_snr = 1:n_snr
    P_tx = Noise_power_sen * SNR_Tx(idx_snr);
    for idx_mc = 1:Mc
        main3_signal_channel_model;
        main4_sensing_process;

        scene = scene + 1;
        RD_stack_data(:,:,scene)   = RD_map_shifted;
        RD_stack_nodata(:,:,scene) = RD_map_noise_no_data_shifted;
        scene_snr(scene) = idx_snr;
        if idx_snr == diag_snr_idx_pass1
            RD_stack_clean_diag(:,:,idx_mc) = RD_map_no_noise_shifted;
            if idx_mc == 1
                RD_ex_clean = double(RD_map_no_noise_shifted);  % Z_rd_prbs
            end
        end

        [eR, eV, hit] = score_branch(RD_map_shifted, N_g_r, N_g_d, N_t_r, N_t_d, ...
            P_fa, peak_select, range_axis, vel_axis, ...
            truth_range, truth_vel, UseGate=USE_GATE, GateR=gate_r, GateV=gate_v, ...
            UseROI=USE_ROI, ROIRange=roi_range, ROIVel=roi_vel);
        errR_conv_data(scene,:) = eR;  errV_conv_data(scene,:) = eV;
        hit_conv_data(scene,:)  = hit;

        [eR, eV, hit] = score_branch(RD_map_noise_no_data_shifted, N_g_r, N_g_d, N_t_r, N_t_d, ...
            P_fa, peak_select, range_axis, vel_axis, ...
            truth_range, truth_vel, UseGate=USE_GATE, GateR=gate_r, GateV=gate_v, ...
            UseROI=USE_ROI, ROIRange=roi_range, ROIVel=roi_vel);
        errR_conv_nodata(scene,:) = eR;  errV_conv_nodata(scene,:) = eV;
        hit_conv_nodata(scene,:)  = hit;
    end
    fprintf('   SNR = %+3d dB done\n', SNR_Tx_db(idx_snr));
end

% =====================================================================
%  5) Batched RDPDNet inference on BOTH stacks
% =====================================================================
fprintf('RDPDNet inference on 2 x %d RD maps (batched)...\n', n_scene);
RD_den_data_batch = inference_run_pdnet( ...
    permute(RD_stack_data, [3 1 2]), string(checkpoint_ml), ...
    PythonExecutable = python_exe_ml, ...
    Device    = string(PDISAC_cfg.inference.device), ...
    BatchSize = PDISAC_cfg.inference.batch_size);
RD_den_nodata_batch = inference_run_pdnet( ...
    permute(RD_stack_nodata, [3 1 2]), string(checkpoint_ml), ...
    PythonExecutable = python_exe_ml, ...
    Device    = string(PDISAC_cfg.inference.device), ...
    BatchSize = PDISAC_cfg.inference.batch_size);

% =====================================================================
%  6) Pass 2: score the ML branches on the denoised RD maps
% =====================================================================
fprintf('Pass 2/2: RDPDNet-denoised CFAR scoring (data & no-data)\n');
errR_ml_data   = nan(n_scene, N_tars);  errV_ml_data   = nan(n_scene, N_tars);
errR_ml_nodata = nan(n_scene, N_tars);  errV_ml_nodata = nan(n_scene, N_tars);
hit_ml_data    = false(n_scene, N_tars);
hit_ml_nodata  = false(n_scene, N_tars);
for s = 1:n_scene
    [eR, eV, hit] = score_branch(squeeze(RD_den_data_batch(s,:,:)), ...
        N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select, range_axis, vel_axis, ...
        truth_range, truth_vel, UseGate=USE_GATE, GateR=gate_r, GateV=gate_v, ...
        UseROI=USE_ROI, ROIRange=roi_range, ROIVel=roi_vel);
    errR_ml_data(s,:) = eR;  errV_ml_data(s,:) = eV;
    hit_ml_data(s,:)  = hit;

    [eR, eV, hit] = score_branch(squeeze(RD_den_nodata_batch(s,:,:)), ...
        N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select, range_axis, vel_axis, ...
        truth_range, truth_vel, UseGate=USE_GATE, GateR=gate_r, GateV=gate_v, ...
        UseROI=USE_ROI, ROIRange=roi_range, ROIVel=roi_vel);
    errR_ml_nodata(s,:) = eR;  errV_ml_nodata(s,:) = eV;
    hit_ml_nodata(s,:)  = hit;
end

% =====================================================================
%  7) Aggregate per-target RMSE/bias per SNR (main5_sensing_analysis style)
%     + bias-adjusted CRLB via func_ana_crlb_rv (bias: conv NO-DATA branch)
% =====================================================================
% Per-target matrices (n_snr x N_tars), exactly as in main5:
%   rmse(idx_snr, j) = sqrt(mean(err(:,j).^2, 'omitnan')) over the Mc trials
%   bias(idx_snr, j) = mean(err(:,j), 'omitnan')
rmse_R_ml_data     = nan(n_snr, N_tars); rmse_v_ml_data     = nan(n_snr, N_tars);
rmse_R_ml_nodata   = nan(n_snr, N_tars); rmse_v_ml_nodata   = nan(n_snr, N_tars);
rmse_R_conv_data   = nan(n_snr, N_tars); rmse_v_conv_data   = nan(n_snr, N_tars);
rmse_R_conv_nodata = nan(n_snr, N_tars); rmse_v_conv_nodata = nan(n_snr, N_tars);
bias_R_conv_nodata = nan(n_snr, N_tars); bias_v_conv_nodata = nan(n_snr, N_tars);
crlb_R_ana         = nan(n_snr, N_tars); crlb_v_ana         = nan(n_snr, N_tars);
crlb_R_ana_adj     = nan(n_snr, N_tars); crlb_v_ana_adj     = nan(n_snr, N_tars);

% Detection probability per SNR (mean over targets and Mc trials)
pd_ml_data     = nan(n_snr,1); pd_ml_nodata   = nan(n_snr,1);
pd_conv_data   = nan(n_snr,1); pd_conv_nodata = nan(n_snr,1);

for idx_snr = 1:n_snr
    rows = (scene_snr == idx_snr);

    rmse_R_ml_data(idx_snr,:)     = sqrt(mean(errR_ml_data(rows,:).^2,     'omitnan'));
    rmse_v_ml_data(idx_snr,:)     = sqrt(mean(errV_ml_data(rows,:).^2,     'omitnan'));
    rmse_R_ml_nodata(idx_snr,:)   = sqrt(mean(errR_ml_nodata(rows,:).^2,   'omitnan'));
    rmse_v_ml_nodata(idx_snr,:)   = sqrt(mean(errV_ml_nodata(rows,:).^2,   'omitnan'));
    rmse_R_conv_data(idx_snr,:)   = sqrt(mean(errR_conv_data(rows,:).^2,   'omitnan'));
    rmse_v_conv_data(idx_snr,:)   = sqrt(mean(errV_conv_data(rows,:).^2,   'omitnan'));
    rmse_R_conv_nodata(idx_snr,:) = sqrt(mean(errR_conv_nodata(rows,:).^2, 'omitnan'));
    rmse_v_conv_nodata(idx_snr,:) = sqrt(mean(errV_conv_nodata(rows,:).^2, 'omitnan'));

    pd_ml_data(idx_snr)     = mean(hit_ml_data(rows,:),     'all');
    pd_ml_nodata(idx_snr)   = mean(hit_ml_nodata(rows,:),   'all');
    pd_conv_data(idx_snr)   = mean(hit_conv_data(rows,:),   'all');
    pd_conv_nodata(idx_snr) = mean(hit_conv_nodata(rows,:), 'all');

    % Empirical bias of the conventional NO-DATA pipeline (per target)
    bias_R_conv_nodata(idx_snr,:) = mean(errR_conv_nodata(rows,:), 'omitnan');
    bias_v_conv_nodata(idx_snr,:) = mean(errV_conv_nodata(rows,:), 'omitnan');

    % Unbiased (theoretical) CRLB per target at this SNR — func_ana_crlb_rv
    P_tx = Noise_power_sen * SNR_Tx(idx_snr);
    [current_crlb_r_ana, current_crlb_v_ana, ~, ~] = func_ana_crlb_rv( ...
        N_tars, Tars_position, Tars_rcs, Tars_vel, ...
        Tx_position, N_tx_ant, Rx_position, N_rx_ant, ...
        P_tx, G_tx, G_rx, Lambda, fc, Noise_power_sen, ...
        T_chip, N_chip, N_block);
    crlb_R_ana(idx_snr,:) = sqrt(current_crlb_r_ana);
    crlb_v_ana(idx_snr,:) = sqrt(current_crlb_v_ana);

    % Bias-adjusted bound, per target (main5 eq.): sqrt(CRLB + bias^2)
    bR = bias_R_conv_nodata(idx_snr,:); bR(isnan(bR)) = 0;
    bV = bias_v_conv_nodata(idx_snr,:); bV(isnan(bV)) = 0;
    crlb_R_ana_adj(idx_snr,:) = sqrt(current_crlb_r_ana(:).' + bR.^2);
    crlb_v_ana_adj(idx_snr,:) = sqrt(current_crlb_v_ana(:).' + bV.^2);

    fprintf(['SNR=%+3d dB | R: ml-d=%.3f ml-nd=%.3f cv-d=%.3f cv-nd=%.3f bnd=%.3f | ' ...
             'V: ml-d=%.3f ml-nd=%.3f cv-d=%.3f cv-nd=%.3f bnd=%.3f\n'], ...
        SNR_Tx_db(idx_snr), ...
        mean(rmse_R_ml_data(idx_snr,:),'omitnan'),     mean(rmse_R_ml_nodata(idx_snr,:),'omitnan'), ...
        mean(rmse_R_conv_data(idx_snr,:),'omitnan'),   mean(rmse_R_conv_nodata(idx_snr,:),'omitnan'), ...
        mean(crlb_R_ana_adj(idx_snr,:),'omitnan'), ...
        mean(rmse_v_ml_data(idx_snr,:),'omitnan'),     mean(rmse_v_ml_nodata(idx_snr,:),'omitnan'), ...
        mean(rmse_v_conv_data(idx_snr,:),'omitnan'),   mean(rmse_v_conv_nodata(idx_snr,:),'omitnan'), ...
        mean(crlb_v_ana_adj(idx_snr,:),'omitnan'));
end

% ---- Mean over the N_tars targets (main5: "Mean RMSE vs SNR") --------------
rmseR_ml_data     = mean(rmse_R_ml_data,     2, 'omitnan');
rmseV_ml_data     = mean(rmse_v_ml_data,     2, 'omitnan');
rmseR_ml_nodata   = mean(rmse_R_ml_nodata,   2, 'omitnan');
rmseV_ml_nodata   = mean(rmse_v_ml_nodata,   2, 'omitnan');
rmseR_conv_data   = mean(rmse_R_conv_data,   2, 'omitnan');
rmseV_conv_data   = mean(rmse_v_conv_data,   2, 'omitnan');
rmseR_conv_nodata = mean(rmse_R_conv_nodata, 2, 'omitnan');
rmseV_conv_nodata = mean(rmse_v_conv_nodata, 2, 'omitnan');
crlbR_adj         = mean(crlb_R_ana_adj,     2, 'omitnan');
crlbV_adj         = mean(crlb_v_ana_adj,     2, 'omitnan');
crlbR_unbiased    = mean(crlb_R_ana,         2, 'omitnan'); %#ok<NASGU> (exported)
crlbV_unbiased    = mean(crlb_v_ana,         2, 'omitnan'); %#ok<NASGU> (exported)

summary = table(SNR_Tx_db(:), ...
    rmseR_ml_data, rmseR_ml_nodata, rmseR_conv_data, rmseR_conv_nodata, crlbR_adj, ...
    rmseV_ml_data, rmseV_ml_nodata, rmseV_conv_data, rmseV_conv_nodata, crlbV_adj, ...
    pd_ml_data, pd_ml_nodata, pd_conv_data, pd_conv_nodata, ...
    'VariableNames', {'SNR_dB', ...
        'RMSE_R_pdnet_data_m','RMSE_R_pdnet_nodata_m', ...
        'RMSE_R_conv_data_m','RMSE_R_conv_nodata_m','sqrtCRLB_adj_R_m', ...
        'RMSE_v_pdnet_data_mps','RMSE_v_pdnet_nodata_mps', ...
        'RMSE_v_conv_data_mps','RMSE_v_conv_nodata_mps','sqrtCRLB_adj_v_mps', ...
        'Pd_pdnet_data','Pd_pdnet_nodata','Pd_conv_data','Pd_conv_nodata'});
csv_path = fullfile(stat_dir, 'statistics_main9b_data_vs_nodata.csv');
writetable(summary, csv_path);
fprintf('\nSummary saved to %s\n', csv_path);

% =====================================================================
%  8) FIGURE 1 — RD heatmaps + AFM mask (representative high-SNR scene)
% =====================================================================

[~, diagnostic_snr_idx] = max(SNR_Tx_db);
diagnostic_scene = find(scene_snr == diagnostic_snr_idx, 1, 'first');

RD_ex_data   = double(RD_stack_data(:,:,diagnostic_scene));
RD_ex_nodata = double(RD_stack_nodata(:,:,diagnostic_scene));

fprintf('RDPDNet + AFM-mask inference for the representative scene...\n');
% M_tars must be built from Z_rd_prbs = the CLEAN map (no noise, no data),
% exactly as in training (dataloader.py: P_rd_prbs = RD_map_no_noise).
[RD_ex_denoised, M_afm_ex, mask_meta] = inference_run_pdnet_mask( ...
    RD_ex_data, RD_ex_clean, string(checkpoint_ml), ...
    PythonExecutable = python_exe_ml, ...
    Device    = string(PDISAC_cfg.inference.device), ...
    BatchSize = 1);
RD_ex_denoised = double(RD_ex_denoised);
% Masks are 2 x H x W (real/imag channels): combine into a complex matrix
% and convert to power dB with the SAME formula as the RD maps:
% 10*log10(abs(.)^2 + eps).
M_afm_display  = mask_to_db(M_afm_ex);                    % H x W, dB
M_mask_display = mask_to_db(mask_meta.M_mask_inference);  % H x W, dB
M_tars_display = mask_to_db(mask_meta.M_tars_inference);  % H x W, dB

% Shared color scale across the three mask panels (dB), like power_limits
mask_limits = [min([M_mask_display(:); M_tars_display(:); M_afm_display(:)]), ...
               max([M_mask_display(:); M_tars_display(:); M_afm_display(:)]) + eps];

pow_data_db     = 10*log10(abs(RD_ex_data).^2     + eps);
pow_nodata_db   = 10*log10(abs(RD_ex_nodata).^2   + eps);
pow_denoised_db = 10*log10(abs(RD_ex_denoised).^2 + eps);
power_limits = [min([pow_data_db(:); pow_nodata_db(:); pow_denoised_db(:)]), ...
                max([pow_data_db(:); pow_nodata_db(:); pow_denoised_db(:)])];

% ---- Figure 1a: RD heatmaps (1 x 3) ----------------------------------------
fig_maps = figure('Name','main9b: RD maps', ...
    'Position',[80 60 1750 520],'Color','w');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
plot_rd_map(pow_data_db, range_axis, vel_axis, truth_range, truth_vel, ...
    roi_range, roi_vel, power_limits, ...
    '(a) $|\mathbf{Z}_{\rm rd}|^2 = |\mathbf{Z}_{\rm rd}^{\rm prbs} + \mathbf{Z}_{\rm rd}^{\rm ran} + \mathbf{N}_{\rm rd}^{\rm mf}|^2$', 'RD power (dB)', parula);

nexttile;
plot_rd_map(pow_nodata_db, range_axis, vel_axis, truth_range, truth_vel, ...
    roi_range, roi_vel, power_limits, ...
    '(b) $|\mathbf{Z}_{\rm rd}|^2 = |\mathbf{Z}_{\rm rd}^{\rm prbs} + \mathbf{N}_{\rm rd}^{\rm mf}|^2$', 'RD power (dB)', parula);

nexttile;
plot_rd_map(pow_denoised_db, range_axis, vel_axis, truth_range, truth_vel, ...
    roi_range, roi_vel, power_limits, ...
    '(c) $|\hat{\mathbf{Z}}_{\rm rd}|^2$', 'RD power (dB)', parula);

savefig(fig_maps, fullfile(fig_dir,'fig_rd_maps.fig'));
exportgraphics(fig_maps, fullfile(fig_dir,'fig_rd_maps.png'),'Resolution',300);

% ---- Figure 1b: masks (1 x 3), same style as fig_rd_maps ------------
fig_masks = figure('Name','main9b: AFM masks', ...
    'Position',[80 640 1750 520],'Color','w');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
plot_mask_map(M_mask_display, range_axis, vel_axis, ...
    roi_range, roi_vel, mask_limits, ...
    '(a) MaskNet output $\mathbf{M}_{\rm mask}$', 'Mask magnitude (dB)', parula);

nexttile;
plot_mask_map(M_tars_display, range_axis, vel_axis, ...
    roi_range, roi_vel, mask_limits, ...
    '(b) Target-protection mask $\mathbf{M}_{\rm tars}$', 'Mask magnitude (dB)', parula);

nexttile;
plot_mask_map(M_afm_display, range_axis, vel_axis, ...
    roi_range, roi_vel, mask_limits, ...
    '(c) $\mathbf{M}_{\rm afm} = \mathbf{M}_{\rm mask} \odot (1 - \mathbf{M}_{\rm tars})$', 'Mask magnitude (dB)', parula);

sgtitle(sprintf('Adversarial frequency masks', ...
    SNR_Tx_db(diagnostic_snr_idx)));
savefig(fig_masks, fullfile(fig_dir,'fig_afm_masks.fig'));
exportgraphics(fig_masks, fullfile(fig_dir,'fig_afm_masks.png'),'Resolution',300);

% =====================================================================
%  9) FIGURES 2 & 3 — RMSE vs SNR (range, velocity), 5 lines each
% =====================================================================
% Okabe-Ito colour-blind-safe palette (same family as main8b)
c_ml_d  = [  0 114 178]/255;   % RDPDNet, data           (blue)
c_ml_nd = [ 86 180 233]/255;   % RDPDNet, no data        (sky blue)
c_cv_d  = [213  94   0]/255;   % conventional, data    (vermillion)
c_cv_nd = [230 159   0]/255;   % conventional, no data (orange)
c_bnd   = [  0   0   0];       % sqrt(CRLB + bias^2)    (black)

% Line-style convention (match main8b): simulation curves = dashed line +
% markers; the theoretical bound = solid line.
LS_SIM = '--';  LW_SIM = 1.7;  MS = 6.5;
LS_BND = '-';   LW_BND = 2.0;

legend_entries = { ...
    'RDPDNet ($\mathbf{Z}_{\rm rd} = \mathbf{Z}_{\rm rd}^{\rm prbs} + \mathbf{Z}_{\rm rd}^{\rm ran} + \mathbf{N}_{\rm rd}^{\rm mf}$)', ...
    'RDPDNet ($\mathbf{Z}_{\rm rd} = \mathbf{Z}_{\rm rd}^{\rm prbs} + \mathbf{N}_{\rm rd}^{\rm mf}$)', ...
    'Conventional ($\mathbf{Z}_{\rm rd} = \mathbf{Z}_{\rm rd}^{\rm prbs} + \mathbf{Z}_{\rm rd}^{\rm ran} + \mathbf{N}_{\rm rd}^{\rm mf}$)', ...
    'Conventional ($\mathbf{Z}_{\rm rd} = \mathbf{Z}_{\rm rd}^{\rm prbs} + \mathbf{N}_{\rm rd}^{\rm mf}$)', ...
    '$\sqrt{\mathrm{CRLB} + b_{\rm bias}^2}$'};

fig_rmse_r = figure('Name','main9b: range RMSE vs SNR', ...
    'Units','inches','Position',[1 1 7 5.2],'Color','w');
ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
semilogy(SNR_Tx_db, rmseR_ml_data,     LS_SIM,'Marker','o','LineWidth',LW_SIM,'Color',c_ml_d, 'MarkerFaceColor',c_ml_d, 'MarkerSize',MS);
semilogy(SNR_Tx_db, rmseR_ml_nodata,   LS_SIM,'Marker','^','LineWidth',LW_SIM,'Color',c_ml_nd,'MarkerFaceColor',c_ml_nd,'MarkerSize',MS);
semilogy(SNR_Tx_db, rmseR_conv_data,   LS_SIM,'Marker','s','LineWidth',LW_SIM,'Color',c_cv_d, 'MarkerFaceColor',c_cv_d, 'MarkerSize',MS);
semilogy(SNR_Tx_db, rmseR_conv_nodata, LS_SIM,'Marker','d','LineWidth',LW_SIM,'Color',c_cv_nd,'MarkerFaceColor',c_cv_nd,'MarkerSize',MS);
semilogy(SNR_Tx_db, crlbR_adj,         LS_BND,'LineWidth',LW_BND,'Color',c_bnd);
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)'); ylabel('Conditional mean range RMSE (m)');
title('Mean range RMSE vs. SNR');
legend(legend_entries, 'Location','northeast','Box','on','FontSize',10);
savefig(fig_rmse_r, fullfile(fig_dir,'fig_rmse_range_vs_snr.fig'));
exportgraphics(fig_rmse_r, fullfile(fig_dir,'fig_rmse_range_vs_snr.png'),'Resolution',300);

fig_rmse_v = figure('Name','main9b: velocity RMSE vs SNR', ...
    'Units','inches','Position',[1 1 7 5.2],'Color','w');
ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
semilogy(SNR_Tx_db, rmseV_ml_data,     LS_SIM,'Marker','o','LineWidth',LW_SIM,'Color',c_ml_d, 'MarkerFaceColor',c_ml_d, 'MarkerSize',MS);
semilogy(SNR_Tx_db, rmseV_ml_nodata,   LS_SIM,'Marker','^','LineWidth',LW_SIM,'Color',c_ml_nd,'MarkerFaceColor',c_ml_nd,'MarkerSize',MS);
semilogy(SNR_Tx_db, rmseV_conv_data,   LS_SIM,'Marker','s','LineWidth',LW_SIM,'Color',c_cv_d, 'MarkerFaceColor',c_cv_d, 'MarkerSize',MS);
semilogy(SNR_Tx_db, rmseV_conv_nodata, LS_SIM,'Marker','d','LineWidth',LW_SIM,'Color',c_cv_nd,'MarkerFaceColor',c_cv_nd,'MarkerSize',MS);
semilogy(SNR_Tx_db, crlbV_adj,         LS_BND,'LineWidth',LW_BND,'Color',c_bnd);
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)'); ylabel('Conditional mean velocity RMSE (m/s)');
title('Mean velocity RMSE vs. SNR');
legend(legend_entries, 'Location','northeast','Box','on','FontSize',10);
savefig(fig_rmse_v, fullfile(fig_dir,'fig_rmse_vel_vs_snr.fig'));
exportgraphics(fig_rmse_v, fullfile(fig_dir,'fig_rmse_vel_vs_snr.png'),'Resolution',300);

% =====================================================================
%  10) FIGURE 4 — Detection probability vs SNR (4 branches)
% =====================================================================
fig_pd = figure('Name','main9b: detection probability vs SNR', ...
    'Units','inches','Position',[1 1 7 5.2],'Color','w');
ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
plot(SNR_Tx_db, pd_ml_data,     '-o', 'LineWidth',LW_SIM,'Color',c_ml_d, 'MarkerFaceColor',c_ml_d, 'MarkerSize',MS);
plot(SNR_Tx_db, pd_ml_nodata,   '-^', 'LineWidth',LW_SIM,'Color',c_ml_nd,'MarkerFaceColor',c_ml_nd,'MarkerSize',MS);
plot(SNR_Tx_db, pd_conv_data,   '--s','LineWidth',LW_SIM,'Color',c_cv_d, 'MarkerFaceColor',c_cv_d, 'MarkerSize',MS);
plot(SNR_Tx_db, pd_conv_nodata, '--d','LineWidth',LW_SIM,'Color',c_cv_nd,'MarkerFaceColor',c_cv_nd,'MarkerSize',MS);
ylim([-0.02 1.02]);
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)'); ylabel('Detection probability $P_d$');
title('Probability that targets appear on the RD map vs. SNR');
legend({'RDPDNet (input: data)','RDPDNet (input: no data)', ...
        'Conventional (data)','Conventional (no data)'}, ...
        'Location','southeast','Box','on','FontSize',10);
savefig(fig_pd, fullfile(fig_dir,'fig_pd_vs_snr.fig'));
exportgraphics(fig_pd, fullfile(fig_dir,'fig_pd_vs_snr.png'),'Resolution',300);

% =====================================================================
%  11) FIGURE 5 — Bias significance vs SNR (main5 style, conv NO-DATA)
%  Bias of the CONVENTIONAL pipeline on the NO-DATA RD map:
%      RD_map_noise_no_data_shifted -> CFAR -> sinc -> (r, v)
%  Residuals pooled across targets per SNR; percentile bootstrap 95% CI on
%  the mean (bias) + z-score z = bias/(std/sqrt(N)). CI excluding 0 (or
%  |z| > 1.96) => statistically significant bias at 95%.
% =====================================================================
n_boot   = 2000;
alpha_ci = 0.05;
rng(2);  % separate reproducible stream for the bias bootstrap

% Pool the conv NO-DATA residuals per SNR (cell array, Mc x N_tars each)
err_R_nodata_store = cell(n_snr, 1);
err_v_nodata_store = cell(n_snr, 1);
for idx_snr = 1:n_snr
    rows = (scene_snr == idx_snr);
    err_R_nodata_store{idx_snr} = errR_conv_nodata(rows,:);
    err_v_nodata_store{idx_snr} = errV_conv_nodata(rows,:);
end

[bias_R_lo, bias_R_hi, bias_R_mid, z_R] = local_bootstrap_mean_ci( ...
    err_R_nodata_store, n_boot, alpha_ci);
[bias_v_lo, bias_v_hi, bias_v_mid, z_v] = local_bootstrap_mean_ci( ...
    err_v_nodata_store, n_boot, alpha_ci);

sig_R = (bias_R_lo > 0) | (bias_R_hi < 0);   % CI excludes zero
sig_v = (bias_v_lo > 0) | (bias_v_hi < 0);

fprintf('\n===== Bias-significance test (conv NO-DATA, pooled over targets) =====\n');
fprintf('%6s | %12s %10s %6s | %12s %10s %6s\n', ...
    'SNR dB', 'Bias_R (m)', 'z_R', 'sig?', 'Bias_v (m/s)', 'z_v', 'sig?');
for idx_snr = 1:n_snr
    fprintf('%6.1f | %12.3e %10.2f %6s | %12.3e %10.2f %6s\n', ...
        SNR_Tx_db(idx_snr), ...
        bias_R_mid(idx_snr), z_R(idx_snr), string(sig_R(idx_snr)), ...
        bias_v_mid(idx_snr), z_v(idx_snr), string(sig_v(idx_snr)));
end

fig_bias_ci = figure('Name','main9b: bias significance vs SNR (conv no-data)', ...
    'Units','inches','Position',[1 1 12 4.8],'Color','w');
tiledlayout(fig_bias_ci, 1, 2, 'TileSpacing','compact','Padding','compact');

nexttile;
ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
xr = SNR_Tx_db(:);
vr = ~isnan(bias_R_lo) & ~isnan(bias_R_hi);
fill([xr(vr); flipud(xr(vr))], [bias_R_lo(vr); flipud(bias_R_hi(vr))], ...
    c_cv_nd, 'FaceAlpha', 0.20, 'EdgeColor', 'none');
plot(xr, bias_R_mid, '-o', 'LineWidth', LW_SIM, 'Color', c_cv_nd, ...
    'MarkerFaceColor', c_cv_nd, 'MarkerSize', MS);
yline(0, '--', 'LineWidth', 1.2, 'Color', c_bnd);
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)'); ylabel('Range bias (m)');
title('Range bias');
legend({'Confidence Interval ','Empirical bias','Zero bias'}, ...
    'Location','best','Box','on','FontSize',10);

nexttile;
ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
vv = ~isnan(bias_v_lo) & ~isnan(bias_v_hi);
fill([xr(vv); flipud(xr(vv))], [bias_v_lo(vv); flipud(bias_v_hi(vv))], ...
    c_cv_nd, 'FaceAlpha', 0.20, 'EdgeColor', 'none');
plot(xr, bias_v_mid, '-o', 'LineWidth', LW_SIM, 'Color', c_cv_nd, ...
    'MarkerFaceColor', c_cv_nd, 'MarkerSize', MS);
yline(0, '--', 'LineWidth', 1.2, 'Color', c_bnd);
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)'); ylabel('Velocity bias (m/s)');
title('Velocity bias');
legend({'Confidence Interval','Empirical bias','Zero bias'}, ...
    'Location','best','Box','on','FontSize',10);

sgtitle('Bias of conventional no-data pipeline vs SNR');
savefig(fig_bias_ci, fullfile(fig_dir,'fig_bias_significance_vs_snr.fig'));
exportgraphics(fig_bias_ci, fullfile(fig_dir,'fig_bias_significance_vs_snr.png'),'Resolution',300);

% =====================================================================
%  12) TABLE — Params, FLOPs/frame, Latency for three pipelines
%      (1) MF + CA-CFAR       : conventional detector, no learnable params
%      (2) RDPDNet (alone)      : the learned denoiser forward pass
%      (3) MF + RDPDNet + CA-CFAR : the full proposed sensing chain
%  The RD-map size is N_chip x N_block; whether the RD map carries data or
%  not does NOT change these numbers (it only changes the pipeline output).
%    - MF+CFAR FLOPs : analytic FFT + power + CA-CFAR estimate (documented).
%    - MF+CFAR latency: measured in MATLAB on a representative frame.
%    - RDPDNet Params/FLOPs/Latency: obtained from the Python backend, reusing
%      models/model.py (count_parameters, count_flops, measure_inference_time)
%      on the configured inference device with warmup — i.e. the pure forward
%      pass, excluding the MATLAB<->Python bridge I/O overhead.
% =====================================================================
fprintf('\nBuilding complexity table (Params / FLOPs/frame / Latency)...\n');

N = N_chip; M = N_block;
n_rep_cpu = 20;

% ---- (A) MF + CA-CFAR analytic FLOPs/frame -------------------------------
fft_flops           = @(L) 5 * L .* log2(L);   % complex FFT, ~5 L log2 L
flops_fasttime_fft  = 2 * M * fft_flops(N);    % fast-time FFT of Rx and Tx
flops_mf_mult       = 6 * N * M;               % complex elementwise MF multiply
flops_doppler_fft   = N * fft_flops(M);        % Doppler FFT across slow-time
flops_fasttime_ifft = M * fft_flops(N);        % IFFT along fast-time
flops_power         = 3 * N * M;               % |Z_rd|^2 power map for CFAR
flops_cfar          = 10 * N * M;              % CA-CFAR (integral-image model)
flops_mfcfar = flops_fasttime_fft + flops_mf_mult + flops_doppler_fft + ...
               flops_fasttime_ifft + flops_power + flops_cfar;

% ---- (B) MF + CA-CFAR latency, measured on a representative frame ---------
RD_repr = double(RD_stack_data(:,:,diagnostic_scene));
t_cfar  = zeros(n_rep_cpu,1);
for r = 1:n_rep_cpu
    tic;
    [~,~,det_pos_t,~] = func_ca_cfar_adaptive_threshold( ...
        RD_repr, N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select);
    if ~isempty(det_pos_t)
        func_sinc_interpolation(RD_repr, det_pos_t, range_axis, vel_axis);
    end
    t_cfar(r) = toc;
end
t_cfar_ms = median(t_cfar) * 1000;

% MF + RD-map formation timing (reconstructed from the last received frame,
% mirroring main4_sensing_process; skipped with NaN if unavailable).
if exist('Data_Rx','var') && exist('Data_Tx','var')
    Rx_mat_t = squeeze(Data_Rx).';
    Tx_mat_t = squeeze(Data_Tx).';
    t_mf = zeros(n_rep_cpu,1);
    for r = 1:n_rep_cpu
        tic;
        Rx_freq = fft(Rx_mat_t);  Tx_freq = fft(Tx_mat_t);
        MFp = Rx_freq .* conj(Tx_freq);
        MFf = fft(MFp, [], 2);  MFf(:,1) = 0;
        RDt = ifft(MFf, [], 1);  RDt = fftshift(RDt, 2); %#ok<NASGU>
        t_mf(r) = toc;
    end
    t_mf_ms = median(t_mf) * 1000;
else
    t_mf_ms = NaN;
end
lat_mfcfar_ms = t_mf_ms + t_cfar_ms;

% ---- (C) RDPDNet Params / FLOPs / Latency from the Python backend -----------
%  Reflects the denoiser currently selected in alg_pdisac/configs/ml.py
%  (RDPDNet for this driver). Reuses the helpers added to models/model.py.
params_pdnet = NaN; flops_pdnet = NaN; lat_pdnet_ms = NaN;
try
    alg_dir_c = fullfile(PDISAC_repo_root, 'alg_pdisac');
    helper_py = fullfile(tempdir, ...
        ['pdnet_complexity_' char(java.util.UUID.randomUUID) '.py']);
    py_src = sprintf([ ...
        'import torch\n' ...
        'from configs.ml import model, device\n' ...
        'from models.model import count_parameters, count_flops, measure_inference_time\n' ...
        'c, h, w = 2, %d, %d\n' ...
        'Y = torch.randn(1, c, h, w).to(device)\n' ...
        'p = count_parameters(model)\n' ...
        '_, f = count_flops(model, {"Z_rd": Y})\n' ...
        'lat = measure_inference_time(model, {"Z_rd": Y}, device=device)\n' ...
        'print("RDPDNET_STATS %%d %%d %%.6f" %% (int(p), int(f), float(lat)))\n'], ...
        N, M);
    fid = fopen(helper_py, 'w');
    fwrite(fid, py_src);
    fclose(fid);

    old_pp = getenv('PYTHONPATH');
    if isempty(old_pp)
        setenv('PYTHONPATH', alg_dir_c);
    else
        setenv('PYTHONPATH', [alg_dir_c pathsep old_pp]);
    end
    cmd = sprintf('"%s" "%s"', char(python_exe_ml), char(helper_py));
    [st, out] = system(cmd);
    setenv('PYTHONPATH', old_pp);
    if isfile(helper_py), delete(helper_py); end

    tok = regexp(out, 'RDPDNET_STATS\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)', 'tokens', 'once');
    if st == 0 && ~isempty(tok)
        params_pdnet = str2double(tok{1});
        flops_pdnet  = str2double(tok{2});
        lat_pdnet_ms = str2double(tok{3});
    else
        warning('main9b:RDPDNetComplexityFailed', ...
            'Could not parse RDPDNet complexity from Python (status %d):\n%s', st, out);
    end
catch ME
    warning('main9b:RDPDNetComplexityError', ...
        'RDPDNet complexity probe failed: %s', ME.message);
end

% ---- (D) Combine and report ----------------------------------------------
params_full = params_pdnet;                 % MF and CFAR add no parameters
flops_full  = flops_mfcfar + flops_pdnet;
lat_full_ms = lat_mfcfar_ms + lat_pdnet_ms;

methods   = {'MF + CA-CFAR'; 'RDPDNet (alone)'; 'MF + RDPDNet + CA-CFAR'};
params_v  = [0;              params_pdnet;    params_full];
flops_g_v = [flops_mfcfar;   flops_pdnet;     flops_full] / 1e9;   % GFLOPs
lat_v     = [lat_mfcfar_ms;  lat_pdnet_ms;    lat_full_ms];        % ms

Tcomplex = table(methods, params_v, flops_g_v, lat_v, ...
    'VariableNames', {'Method','Params','FLOPs_per_frame_G','Latency_ms'});
complex_csv = fullfile(stat_dir, 'statistics_main9b_complexity.csv');
writetable(Tcomplex, complex_csv);

fprintf('\n%s\n', repmat('=',1,74));
fprintf('  Complexity  (RD map %d x %d; independent of embedded data)\n', N, M);
fprintf('%s\n', repmat('=',1,74));
fprintf('%-24s %12s %16s %13s\n', 'Method', 'Params', 'FLOPs/frame', 'Latency');
fprintf('%s\n', repmat('-',1,74));
fprintf('%-24s %12s %13.3f G %10.3f ms\n', 'MF + CA-CFAR', ...
    '0', flops_mfcfar/1e9, lat_mfcfar_ms);
fprintf('%-24s %12s %13.3f G %10.3f ms\n', 'RDPDNet (alone)', ...
    num2str(params_pdnet), flops_pdnet/1e9, lat_pdnet_ms);
fprintf('%-24s %12s %13.3f G %10.3f ms\n', 'MF + RDPDNet + CA-CFAR', ...
    num2str(params_full), flops_full/1e9, lat_full_ms);
fprintf('%s\n', repmat('=',1,74));
fprintf(['Notes: MF+CFAR FLOPs are an analytic FFT+CFAR estimate; MF+CFAR latency\n' ...
         'is measured in MATLAB; RDPDNet Params/FLOPs/Latency come from the Python\n' ...
         'backend (warmup forward pass, device=%s), excluding bridge I/O overhead.\n'], ...
         string(PDISAC_cfg.inference.device));
fprintf('Complexity table saved to %s\n', complex_csv);

% =====================================================================
%  13) FIGURE 6 — RDPDNet distribution surfaces (3-D, Monte-Carlo averaged)
%  Exports every distribution produced by model.distribution() through the
%  Python bridge (alg_pdisac/scripts/distribution_pdnet_mat.py):
%      p(Z_rd)                observed (data-embedded) RD map
%      p(Z_rd^prbs)           clean / data-free reference
%      p(z_l | z_{l-1})       bottom-up (encoder) latents
%      p(z_L)                 top latent
%      p(z_{l-1} | z_l)       top-down (decoder) latents
%      p(Zhat_rd)             reconstruction
%  Each map is averaged over the Monte-Carlo trials and over the channel
%  dimension (as in vis_dis.py, aggregation="mean"), then rendered as a 3-D
%  surface in dB, in the style of the reference Python plot. Averaging over
%  many random-scene trials fills in the whole FOV and yields a smooth
%  surface rather than the isolated spikes of a single trial.
%  NOTE: the maps live in the network's normalized space, so the dB values
%  are relative, not absolute receive power.
% =====================================================================
DIST_MAX_TRIALS = 100;   % cap on the number of trials pushed through the bridge
try
    rows_diag = find(scene_snr == diag_snr_idx_pass1);
    n_dist_trials = min([DIST_MAX_TRIALS, numel(rows_diag), size(RD_stack_clean_diag,3)]);
    if n_dist_trials < 1
        error('main9b:NoDistributionScenes', 'No diagnostic-SNR scenes available.');
    end

    RD_dist_noisy = RD_stack_data(:,:,rows_diag(1:n_dist_trials));   % Z_rd
    RD_dist_clean = RD_stack_clean_diag(:,:,1:n_dist_trials);        % Z_rd^prbs

    fprintf('\nRDPDNet distribution at SNR = %+d dB...\n', ...
        n_dist_trials, SNR_Tx_db(diag_snr_idx_pass1));

    [dist_maps, dist_labels] = inference_run_pdnet_distribution( ...
        permute(RD_dist_noisy, [3 1 2]), permute(RD_dist_clean, [3 1 2]), ...
        string(checkpoint_ml), ...
        PythonExecutable = python_exe_ml, ...
        Device    = string(PDISAC_cfg.inference.device), ...
        BatchSize = PDISAC_cfg.inference.batch_size);

    n_dist_maps = numel(dist_labels);
    fprintf('  received %d distribution maps (%d x %d each).\n', ...
        n_dist_maps, size(dist_maps,2), size(dist_maps,3));

    % ---- one 3-D surface per distribution -------------------------------
    for kd = 1:n_dist_maps
        map_kd  = squeeze(dist_maps(kd,:,:));            % N_chip x N_block
        surf_db = 10*log10(abs(map_kd).^2 + eps);        % relative power (dB)

        fig_kd = figure('Name', sprintf('main9b: distribution %d', kd), ...
            'Units','inches','Position',[1 1 7.2 5.4],'Color','w');
        ax_kd = gca;
        s_kd = surf(ax_kd, vel_axis, range_axis, surf_db);
        s_kd.EdgeColor = 'none';
        shading(ax_kd,'interp'); colormap(ax_kd, parula);
        box(ax_kd,'on'); grid(ax_kd,'on'); view(ax_kd, 45, 30);
        xlim(ax_kd, sort(roi_vel)); ylim(ax_kd, sort(roi_range));
        xlabel(ax_kd, 'Radial velocity (m/s)');
        ylabel(ax_kd, 'Range (m)');
        zlabel(ax_kd, 'Avg. magnitude (dB)');
        title(ax_kd, sprintf('%s', dist_labels{kd}));
        cb_kd = colorbar(ax_kd); cb_kd.Label.String = 'Avg. magnitude (dB)';

        fname_kd = sprintf('fig_dist_%02d', kd);
        savefig(fig_kd, fullfile(fig_dir,[fname_kd '.fig']));
        exportgraphics(fig_kd, fullfile(fig_dir,[fname_kd '.png']),'Resolution',300);
        fprintf('    %-18s <- %s\n', [fname_kd '.png'], dist_labels{kd});
    end

    % ---- combined overview (all distributions in one tiled figure) ------
    fig_dist_all = figure('Name','main9b: RDPDNet distributions (overview)', ...
        'Units','inches','Position',[1 1 15 9],'Color','w');
    tl_dist = tiledlayout(fig_dist_all,'flow','TileSpacing','compact','Padding','compact');
    for kd = 1:n_dist_maps
        map_kd  = squeeze(dist_maps(kd,:,:));
        surf_db = 10*log10(abs(map_kd).^2 + eps);
        ax_kd = nexttile(tl_dist);
        s_kd = surf(ax_kd, vel_axis, range_axis, surf_db);
        s_kd.EdgeColor = 'none';
        shading(ax_kd,'interp'); colormap(ax_kd, parula);
        box(ax_kd,'on'); grid(ax_kd,'on'); view(ax_kd, 45, 30);
        xlim(ax_kd, sort(roi_vel)); ylim(ax_kd, sort(roi_range));
        xlabel(ax_kd,'Vel. (m/s)'); ylabel(ax_kd,'Range (m)'); zlabel(ax_kd,'dB');
        title(ax_kd, dist_labels{kd});
        set(ax_kd,'FontSize',9);
    end
    title(tl_dist, sprintf('RDPDNet distributions, mean over %d trials at SNR $=%+d$ dB', ...
        n_dist_trials, SNR_Tx_db(diag_snr_idx_pass1)));
    savefig(fig_dist_all, fullfile(fig_dir,'fig_dist_overview.fig'));
    exportgraphics(fig_dist_all, fullfile(fig_dir,'fig_dist_overview.png'),'Resolution',300);

    fprintf('  distribution figures saved to %s\n', fig_dir);
catch ME_dist
    warning('main9b:DistributionPlotFailed', ...
        'RDPDNet distribution figures skipped: %s', ME_dist.message);
end

fprintf('main9b complete. Figures in %s\n', fig_dir);

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function s = tf(b)
    if b, s = 'found'; else, s = 'MISSING'; end
end

% Same style as plot_rd_map (parula/dB, axes, colorbar, ROI limits) but
% WITHOUT the truth markers: these panels show masks, not detections.
function plot_mask_map(map_values, range_axis, vel_axis, ...
        roi_range, roi_vel, color_limits, plot_title, cbar_label, cmap)
    imagesc(vel_axis, range_axis, map_values);
    axis xy; colormap(gca, cmap); clim(color_limits);
    xlim(sort(roi_vel)); ylim(sort(roi_range)); grid on; box on;
    xlabel('Radial velocity (m/s)'); ylabel('Range (m)'); title(plot_title);
    cb = colorbar; cb.Label.String = cbar_label;
end

function plot_rd_map(map_values, range_axis, vel_axis, truth_range, truth_vel, ...
        roi_range, roi_vel, color_limits, plot_title, cbar_label, cmap)
    imagesc(vel_axis, range_axis, map_values);
    axis xy; hold on; colormap(gca, cmap); clim(color_limits);
    plot(truth_vel, truth_range, 'rx','MarkerSize',10,'LineWidth',2, ...
        'DisplayName','Truth');
    xlim(sort(roi_vel)); ylim(sort(roi_range)); grid on; box on;
    xlabel('Radial velocity (m/s)'); ylabel('Range (m)'); title(plot_title);
    cb = colorbar; cb.Label.String = cbar_label;
    legend('Location','best');
end

function db = mask_to_db(mask_2hw)
    % Complex from the 2 channels, then power dB exactly like the RD maps:
    % 10*log10(abs(.)^2 + eps)
    m = double(mask_2hw);
    z = squeeze(m(1,:,:)) + 1i*squeeze(m(2,:,:));
    db = 10*log10(abs(z).^2 + eps);
end

% Estimator back-end: for each target,
% independently take the nearest detection among ALL detections (no removal,
% no one-to-one constraint), with dist2 = dR.^2 + dV.^2.
% Options:
%   UseROI  : discard detections outside ROIRange/ROIVel before association.
%   UseGate : accept the nearest detection only if |dR| <= GateR and
%             |dV| <= GateV; otherwise the target is a MISS (hit=false,
%             residuals NaN -> excluded from RMSE/bias, miss in Pd).
% With UseGate=false and UseROI=false, no gating or ROI filtering is applied.
function [errR, errV, hit] = score_branch(RD_map, N_g_r, N_g_d, N_t_r, N_t_d, ...
        P_fa, peak_select, range_axis, vel_axis, truth_range, truth_vel, opts)
    arguments
        RD_map
        N_g_r; N_g_d; N_t_r; N_t_d
        P_fa; peak_select
        range_axis; vel_axis
        truth_range; truth_vel
        opts.UseGate (1,1) logical = true
        opts.GateR   (1,1) double  = Inf
        opts.GateV   (1,1) double  = Inf
        opts.UseROI  (1,1) logical = false
        opts.ROIRange = []
        opts.ROIVel   = []
    end

    N_tars = numel(truth_range);
    errR = nan(N_tars,1); errV = nan(N_tars,1); hit = false(N_tars,1);

    [~,~,det_pos,~] = func_ca_cfar_adaptive_threshold( ...
        RD_map, N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select);
    if isempty(det_pos), return; end

    [det_range, det_vel] = func_sinc_interpolation( ...
        RD_map, det_pos, range_axis, vel_axis);
    det_range = det_range(:);
    det_vel   = det_vel(:);

    if opts.UseROI && ~isempty(opts.ROIRange) && ~isempty(opts.ROIVel)
        in_roi = det_range >= min(opts.ROIRange) & det_range <= max(opts.ROIRange) & ...
                 det_vel   >= min(opts.ROIVel)   & det_vel   <= max(opts.ROIVel);
        det_range = det_range(in_roi);
        det_vel   = det_vel(in_roi);
        if isempty(det_range), return; end
    end

    for j = 1:N_tars
        dR = det_range - truth_range(j);
        dV = det_vel   - truth_vel(j);
        dist2 = dR.^2 + dV.^2;          % distance (unnormalized)
        [~, a] = min(dist2);

        if opts.UseGate && (abs(dR(a)) > opts.GateR || abs(dV(a)) > opts.GateV)
            continue;                    % miss: residuals stay NaN
        end
        hit(j)  = true;
        errR(j) = dR(a);
        errV(j) = dV(a);
    end
end

function [lo, hi, mid, zscore] = local_bootstrap_mean_ci(err_store, n_boot, alpha_ci)
% Percentile bootstrap CI on the MEAN (bias) of the residuals, pooling
% trials across targets at each SNR point (copied from
% main5_sensing_analysis). Also returns the z-score
% z = mean / (std/sqrt(n)); |z| > 1.96 => bias significant at 95%.
    n_snr = numel(err_store);
    lo     = nan(n_snr, 1);
    hi     = nan(n_snr, 1);
    mid    = nan(n_snr, 1);
    zscore = nan(n_snr, 1);
    for idx_snr = 1:n_snr
        err_vec = err_store{idx_snr}(:);
        err_vec = err_vec(~isnan(err_vec));
        n = numel(err_vec);
        if n == 0, continue; end

        boot_mean = zeros(n_boot, 1);
        for b = 1:n_boot
            samp = err_vec(randi(n, n, 1));
            boot_mean(b) = mean(samp);
        end
        boot_mean_sorted = sort(boot_mean);
        lo_idx = max(1, round(n_boot * (alpha_ci/2)));
        hi_idx = min(n_boot, round(n_boot * (1 - alpha_ci/2)));

        lo(idx_snr)  = boot_mean_sorted(lo_idx);
        hi(idx_snr)  = boot_mean_sorted(hi_idx);
        mid(idx_snr) = mean(err_vec);
        sd = std(err_vec, 0);   % sample std (/(n-1))
        if sd > 0
            zscore(idx_snr) = mean(err_vec) / (sd / sqrt(n));
        end
    end
end
