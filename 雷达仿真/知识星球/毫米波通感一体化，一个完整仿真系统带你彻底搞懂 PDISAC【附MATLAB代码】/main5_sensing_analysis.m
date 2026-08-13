% SENSING_ANALYSIS_B  (consolidated Group 5: sensing analysis and plots)
% -------------------------------------------------------------------------
% CRLB (analytical) vs MLE/MF (simulation) validation sweep. 
%
% *** ENHANCEMENT over main5_sensing_analysis ***
% This script incorporates the estimator bias into the 
% CRLB comparison. It computes the empirical bias for the CA-CFAR pipeline 
% and the MLE estimator, and plots RMSE against sqrt(CRLB_unbiased + bias^2).
% This provides a fairer comparison for estimators bounded by their own bias.

clear; clc; close all;

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
stat_dir = fullfile(fileparts(mfilename('fullpath')), 'exported_statistics');
if ~exist(stat_dir,'dir'), mkdir(stat_dir); end

NO_SIGNAL_PLOT = true;   
NO_RADAR_PLOT  = true;   
NO_CFAR_PLOT   = true;   

rng("default");
TOPO_NO_PLOT=false; main2_topology;
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

P_tx = Noise_power_sen * SNR_Tx(length(SNR_Tx));
main3_signal_channel_model;
main4_sensing_process;

Mc = sensing_statistics_cfg.monte_carlo;

rmse_R_cfar = nan(length(SNR_Tx), N_tars);
rmse_v_cfar = nan(length(SNR_Tx), N_tars);
bias_R_cfar = nan(length(SNR_Tx), N_tars);
bias_v_cfar = nan(length(SNR_Tx), N_tars);

rmse_R_mle  = nan(length(SNR_Tx), N_tars);
rmse_v_mle  = nan(length(SNR_Tx), N_tars);
bias_R_mle  = nan(length(SNR_Tx), N_tars);
bias_v_mle  = nan(length(SNR_Tx), N_tars);

% Empirical variance (about the empirical mean) of the MLE residuals; used to
% verify the MSE = Var + Bias^2 identity and to separate the two terms vs SNR.
var_R_mle   = nan(length(SNR_Tx), N_tars);
var_v_mle   = nan(length(SNR_Tx), N_tars);

crlb_R_ana  = nan(length(SNR_Tx), N_tars);
crlb_v_ana  = nan(length(SNR_Tx), N_tars);

crlb_R_ana_adj_cfar = nan(length(SNR_Tx), N_tars);
crlb_v_ana_adj_cfar = nan(length(SNR_Tx), N_tars);
crlb_R_ana_adj_mle  = nan(length(SNR_Tx), N_tars);
crlb_v_ana_adj_mle  = nan(length(SNR_Tx), N_tars);

pd_per = nan(length(SNR_Tx), N_tars);

% Raw per-trial velocity errors kept per SNR (needed later to bootstrap a
% confidence interval on the numerical/simulated RMSE_v).
all_err_v_mle_store = cell(length(SNR_Tx), 1);
all_err_R_mle_store = cell(length(SNR_Tx), 1);

% Per-target and joint FIM captured once (at the first SNR point) for display.
F_tars_display  = {};
F_joint_display = [];

n_rows_total = length(SNR_Tx) * Mc * N_tars;
col_names = { ...
    'Tar_X','Tar_Y','Tar_Z', ...
    'Tar_Vx','Tar_Vy','Tar_Vz', ...
    'Tar_RCS', ...
    'SNR_dB', ...
    'True_Range_m','Est_Range_cfar_m', ...
    'True_Vel_mps','Est_Vel__cfar_mps', ...
    'CRLB_Range_ana_m', 'RMSE_Range_mle_m', ...
    'CRLB_Vel_ana_mps', 'RMSE_Vel_mle_mps', ...
    'MC_Index'};
results_buf = nan(n_rows_total, numel(col_names));
row_ptr = 1;

for idx_snr = 1:length(SNR_Tx)
    snr_tx = SNR_Tx(idx_snr);
    P_tx = Noise_power_sen * snr_tx;
    
    est_R_cfar = nan(Mc, N_tars);
    est_v_cfar = nan(Mc, N_tars);
    est_R_mle = nan(Mc, N_tars);
    est_v_mle = nan(Mc, N_tars);
    all_err_R_cfar = nan(Mc, N_tars);
    all_err_v_cfar = nan(Mc, N_tars);
    all_err_R_mle = nan(Mc, N_tars);
    all_err_v_mle = nan(Mc, N_tars);
    det_count = zeros(1, N_tars);

    for idx_mc = 1:Mc
        main3_signal_channel_model;
        main4_sensing_process;

        range_res  = c / (2 * B);
        range_axis = (0 : N_chip-1) * range_res;
        
        PRF = 1 / T_pmcw; 
        fd_axis  = (-N_block/2 : N_block/2-1) * (PRF / N_block);
        vel_axis = -(Lambda / 2) * fd_axis;

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

        truth_range = sqrt(sum((Tars_position - Tx_position).^2, 1)).';
        unit_vec    = (Tars_position - Tx_position) ./ ...
                      (sqrt(sum((Tars_position - Tx_position).^2, 1)) + eps);
        truth_vel   = sum(Tars_vel .* unit_vec, 1).';
        N_tars     = length(truth_range);

        K_det = size(detected_positions, 1);

        if K_det > 0
            [det_range, det_vel] = func_sinc_interpolation( ...
                RD_map_shifted, detected_positions, range_axis, vel_axis);
        end
        
        assoc     = nan(N_tars, 1);   
        err_R_vec_cfar = nan(N_tars, 1);   
        err_v_vec_cfar = nan(N_tars, 1);   
        
        roi_range = Region_of_interest(1, :);  
        roi_vel   = Region_of_interest(2, :);  
        if K_det > 0
            range_span = max(diff(roi_range), eps);
            vel_span   = max(diff(roi_vel),   eps);
            for j = 1 : N_tars
                dist2 = ((det_range - truth_range(j)) / range_span).^2 + ...
                        ((det_vel   - truth_vel(j)  ) / vel_span  ).^2;
                [~, assoc(j)] = min(dist2);
                err_R_vec_cfar(j)  = det_range(assoc(j)) - truth_range(j);
                err_v_vec_cfar(j)  = det_vel(assoc(j))   - truth_vel(j);
                det_count(j)    = det_count(j) + 1;
            end
        end
        
        if K_det > 0
            est_R_cfar(idx_mc, :) = det_range(assoc);
            est_v_cfar(idx_mc, :) = det_vel(assoc);
        end
        all_err_R_cfar(idx_mc, :) = err_R_vec_cfar;
        all_err_v_cfar(idx_mc, :) = err_v_vec_cfar;
        
        %% MLE estimator
        err_R_vec_mle = nan(N_tars, 1);   
        err_v_vec_mle = nan(N_tars, 1);   
        
        [current_est_R_mle, current_est_v_mle] = func_mle_estimation( ...
                    Data_Rx_no_data, Data_Tx_no_data, N_tars, ...
                    Tars_position, Tars_rcs, Tars_vel, ...
                    Tx_position, N_tx_ant, Rx_position, N_rx_ant, ...
                    P_tx, G_tx, G_rx, Lambda, fc, Noise_power_sen, ...
                    T_chip, N_chip, N_block);

        for j = 1 : N_tars
            err_R_vec_mle(j)  = current_est_R_mle(j) - truth_range(j);
            err_v_vec_mle(j)  = current_est_v_mle(j)  - truth_vel(j);
        end 

        all_err_R_mle(idx_mc, :) = err_R_vec_mle;
        all_err_v_mle(idx_mc, :) = err_v_vec_mle;

        for j = 1 : N_tars
            est_r = nan; est_v = nan;
            if K_det > 0 && ~isnan(assoc(j))
                est_r = det_range(assoc(j));
                est_v = det_vel(assoc(j));
            end
            results_buf(row_ptr, :) = [ ...
                Tars_position(1,j), Tars_position(2,j), Tars_position(3,j), ...
                Tars_vel(1,j),      Tars_vel(2,j),      Tars_vel(3,j),      ...
                Tars_rcs(j),                                                  ...
                SNR_Tx_db(idx_snr),                                           ...
                truth_range(j),     est_r,                         ...
                truth_vel(j),       est_v,                         ...
                nan, nan,                                                      ... 
                nan, nan,                                                      ... 
                idx_mc];
            row_ptr = row_ptr + 1;
        end
    end  

    all_err_v_mle_store{idx_snr} = all_err_v_mle;
    all_err_R_mle_store{idx_snr} = all_err_R_mle;

    rmse_R_cfar(idx_snr, :)   = sqrt(mean(all_err_R_cfar.^2, 'omitnan'));
    rmse_v_cfar(idx_snr, :)   = sqrt(mean(all_err_v_cfar.^2, 'omitnan'));
    bias_R_cfar(idx_snr, :)   = mean(all_err_R_cfar, 'omitnan');
    bias_v_cfar(idx_snr, :)   = mean(all_err_v_cfar, 'omitnan');

    rmse_R_mle(idx_snr, :)    = sqrt(mean(all_err_R_mle.^2, 'omitnan'));
    rmse_v_mle(idx_snr, :)    = sqrt(mean(all_err_v_mle.^2, 'omitnan'));
    bias_R_mle(idx_snr, :)    = mean(all_err_R_mle, 'omitnan');
    bias_v_mle(idx_snr, :)    = mean(all_err_v_mle, 'omitnan');

    % Empirical variance about the empirical mean (population form, /Mc), so
    % that Var + Bias^2 == MSE == mean(err.^2) holds exactly per target.
    var_R_mle(idx_snr, :)     = mean((all_err_R_mle - bias_R_mle(idx_snr, :)).^2, 'omitnan');
    var_v_mle(idx_snr, :)     = mean((all_err_v_mle - bias_v_mle(idx_snr, :)).^2, 'omitnan');

    pd_per(idx_snr, :) = det_count / Mc;

    [current_crlb_r_ana, current_crlb_v_ana, F_tars, F_joint] = func_ana_crlb_rv( ...
        N_tars, Tars_position, Tars_rcs, Tars_vel, ...
        Tx_position, N_tx_ant, Rx_position, N_rx_ant, ...
        P_tx, G_tx, G_rx, Lambda, fc, Noise_power_sen, ...
        T_chip, N_chip, N_block);

    if idx_snr == 1
        F_tars_display  = F_tars;
        F_joint_display = F_joint;
        for m = 1:N_tars
            fprintf('FIM target %d (SNR = %.1f dB):\n', m, SNR_Tx_db(idx_snr));
            disp(F_tars{m});
        end
        fprintf('Joint FIM (block-diagonal, %dx%d):\n', size(F_joint,1), size(F_joint,2));
        disp(F_joint);
    end

    crlb_R_ana(idx_snr, :) = sqrt(current_crlb_r_ana);
    crlb_v_ana(idx_snr, :) = sqrt(current_crlb_v_ana);

    crlb_R_ana_adj_cfar(idx_snr, :) = sqrt(current_crlb_r_ana(:).' + bias_R_cfar(idx_snr, :).^2);
    crlb_v_ana_adj_cfar(idx_snr, :) = sqrt(current_crlb_v_ana(:).' + bias_v_cfar(idx_snr, :).^2);

    crlb_R_ana_adj_mle(idx_snr, :) = sqrt(current_crlb_r_ana(:).' + bias_R_mle(idx_snr, :).^2);
    crlb_v_ana_adj_mle(idx_snr, :) = sqrt(current_crlb_v_ana(:).' + bias_v_mle(idx_snr, :).^2);

    rows_this_snr = (row_ptr - Mc*N_tars) : (row_ptr - 1);
    for mc_i = 1 : Mc
        for j = 1 : N_tars
            r = rows_this_snr(1) + (mc_i-1)*N_tars + (j-1);
            results_buf(r, 13) = crlb_R_ana(idx_snr, j);  
            results_buf(r, 14) = rmse_R_mle(idx_snr, j);  
            results_buf(r, 15) = crlb_v_ana(idx_snr, j);  
            results_buf(r, 16) = rmse_v_mle(idx_snr, j);  
        end
    end

    fprintf('SNR=%+3d dB | RMSE_R CFAR=%.3f m (adjCRLB=%.3f) MLE=%.3f m (adjCRLB=%.3f) | Pd=%.2f\n', ...
        SNR_Tx_db(idx_snr), ...
        mean(rmse_R_cfar(idx_snr,:),'omitnan'), mean(crlb_R_ana_adj_cfar(idx_snr, :),'omitnan'), ...
        mean(rmse_R_mle(idx_snr,:),'omitnan'), mean(crlb_R_ana_adj_mle(idx_snr, :),'omitnan'), ...
        mean(pd_per(idx_snr,:)));
end

results_table = array2table(results_buf, 'VariableNames', col_names);
csv_filename  = fullfile(stat_dir, 'statistics_sensing_biased_crlb.csv');
writetable(results_table, csv_filename);
fprintf('\nResults saved to %s  (%d rows x %d cols)\n', ...
    csv_filename, height(results_table), width(results_table));

% =====================================================================
%  Mean RMSE vs SNR  (averaged over the N_tars targets)
% =====================================================================
mean_rmse_R_cfar     = mean(rmse_R_cfar,     2, 'omitnan');
mean_rmse_v_cfar     = mean(rmse_v_cfar,     2, 'omitnan');
mean_rmse_R_mle      = mean(rmse_R_mle,      2, 'omitnan');
mean_rmse_v_mle      = mean(rmse_v_mle,      2, 'omitnan');

mean_crlb_R_ana = mean(crlb_R_ana, 2, 'omitnan');   % unbiased CRLB (theoretical line)
mean_crlb_v_ana = mean(crlb_v_ana, 2, 'omitnan');   % unbiased CRLB (theoretical line)

mean_crlb_R_ana_adj_cfar = mean(crlb_R_ana_adj_cfar, 2, 'omitnan');
mean_crlb_v_ana_adj_cfar = mean(crlb_v_ana_adj_cfar, 2, 'omitnan');
mean_crlb_R_ana_adj_mle  = mean(crlb_R_ana_adj_mle, 2, 'omitnan');
mean_crlb_v_ana_adj_mle  = mean(crlb_v_ana_adj_mle, 2, 'omitnan');

c_cfar = [0.85 0.33 0.10];
c_mle  = [0.00 0.45 0.74];

fig_mean_rmse = figure('Name','Mean RMSE vs SNR (Biased CRLB)', ...
    'Position',[120 120 1400 520],'Color','w');

subplot(1,2,1);
semilogy(SNR_Tx_db, mean_rmse_R_cfar,     '--s','LineWidth',1.6,'Color',c_cfar, 'MarkerFaceColor',c_cfar); hold on;
semilogy(SNR_Tx_db, mean_crlb_R_ana_adj_cfar, ':s','LineWidth',1.2,'Color',c_cfar.*0.7);
semilogy(SNR_Tx_db, mean_rmse_R_mle,     '-o','LineWidth',2.0,'Color',c_mle, 'MarkerFaceColor',c_mle);
semilogy(SNR_Tx_db, mean_crlb_R_ana_adj_mle, ':o','LineWidth',1.2,'Color',c_mle.*0.7);
grid on; box on;
xlabel('SNR (dB)'); ylabel('Mean RMSE range (m)');
title('Mean Range RMSE vs SNR');
legend({'RMSE (CFAR)','$\sqrt{\mathrm{CRLB} + b_{cfar}^2}$', ...
        'RMSE (MLE)','$\sqrt{\mathrm{CRLB} + b_{mle}^2}$'}, 'Location','best');

subplot(1,2,2);
semilogy(SNR_Tx_db, mean_rmse_v_cfar,     '--s','LineWidth',1.6,'Color',c_cfar, 'MarkerFaceColor',c_cfar); hold on;
semilogy(SNR_Tx_db, mean_crlb_v_ana_adj_cfar, ':s','LineWidth',1.2,'Color',c_cfar.*0.7);
semilogy(SNR_Tx_db, mean_rmse_v_mle,     '-o','LineWidth',2.0,'Color',c_mle, 'MarkerFaceColor',c_mle);
semilogy(SNR_Tx_db, mean_crlb_v_ana_adj_mle, ':o','LineWidth',1.2,'Color',c_mle.*0.7);
grid on; box on;
xlabel('SNR (dB)'); ylabel('Mean RMSE velocity (m/s)');
title('Mean Velocity RMSE vs SNR');
legend({'RMSE (CFAR)','$\sqrt{\mathrm{CRLB} + b_{cfar}^2}$', ...
        'RMSE (MLE)','$\sqrt{\mathrm{CRLB} + b_{mle}^2}$'}, 'Location','best');

sgtitle('Mean Estimation RMSE vs SNR (Bias-Adjusted CRLB)');

% Export
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
savefig(fig_mean_rmse, fullfile(fig_dir,'fig_sensing_mean_rmse_vs_snr_biased_crlb.fig'));
exportgraphics(fig_mean_rmse, fullfile(fig_dir,'fig_sensing_mean_rmse_vs_snr_biased_crlb.png'),'Resolution',300);

% =====================================================================
%  FIM visualization: per-target 2x2 blocks + overall block-diagonal
%  joint FIM (captured at the first SNR point, F_tars_display/F_joint_display).
% =====================================================================
fig_fim = figure('Name','Fisher Information Matrices', ...
    'Position',[100 100 260*(N_tars+1) 300],'Color','w');

for m = 1:N_tars
    subplot(1, N_tars+1, m);
    imagesc(abs(F_tars_display{m})); axis square; colorbar;
    set(gca, 'XTick', 1:2, 'XTickLabel', {'r','v'}, ...
             'YTick', 1:2, 'YTickLabel', {'r','v'});
    title(sprintf('$\\mathbf{F}(\\vec{\\psi}^{(%d)})$', m), 'Interpreter','latex');
    for a = 1:2
        for b = 1:2
            text(b, a, sprintf('%.2e', F_tars_display{m}(a,b)), ...
                'HorizontalAlignment','center', 'Color','w', 'FontSize',8);
        end
    end
end

subplot(1, N_tars+1, N_tars+1);
imagesc(log10(abs(F_joint_display) + eps)); axis square; colorbar;
title('$\log_{10}|\mathbf{F}(\vec{\Psi})|$ (block-diagonal)', 'Interpreter','latex');
xlabel('Stacked index'); ylabel('Stacked index');

sgtitle('Fisher Information Matrices: Individual Targets vs. Joint (Block-Diagonal)');

savefig(fig_fim, fullfile(fig_dir,'fig_fim_individual_and_joint.fig'));
exportgraphics(fig_fim, fullfile(fig_dir,'fig_fim_individual_and_joint.png'),'Resolution',300);

% =====================================================================
%  eq_crlb_v ONLY: theoretical bound (analytical, line) vs. numerical/
%  simulated result (bootstrap 95% confidence interval, shaded band).
%  Pools trials across targets at each SNR point; resamples with
%  replacement to build a percentile CI on the RMSE_v statistic.
% =====================================================================
n_boot   = 2000;
alpha_ci = 0.05;               % 95% CI

rng(1);  % reproducible bootstrap resampling
[rmse_v_lo, rmse_v_hi, rmse_v_mid] = local_bootstrap_rmse_ci( ...
    all_err_v_mle_store, n_boot, alpha_ci);
[rmse_R_lo, rmse_R_hi, rmse_R_mid] = local_bootstrap_rmse_ci( ...
    all_err_R_mle_store, n_boot, alpha_ci);

% ---- Velocity (eq_crlb_v) ----
fig_crlb_v_ci = figure('Name','Velocity CRLB bound: theory (line) vs. simulation (95% CI)', ...
    'Position',[160 160 800 520],'Color','w');

x  = SNR_Tx_db(:);
lo = rmse_v_lo;
hi = rmse_v_hi;
valid = ~isnan(lo) & ~isnan(hi);

fill([x(valid); flipud(x(valid))], [lo(valid); flipud(hi(valid))], ...
    c_mle, 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
semilogy(x, rmse_v_mid, '-o', 'LineWidth', 2.0, 'Color', c_mle, 'MarkerFaceColor', c_mle);
semilogy(x, mean_crlb_v_ana, '-k', 'LineWidth', 2.2);
set(gca, 'YScale', 'log');
grid on; box on;
xlabel('SNR (dB)'); ylabel('$\sqrt{\mathrm{CRLB}_v}$ / RMSE$_v$ (m/s)', 'Interpreter','latex');
title('Velocity CRLB Bound (eq\_crlb\_v): Theoretical Line vs. Numerical 95% CI');
legend({'95% CI (bootstrap, MLE)', 'Mean RMSE (MLE, numerical)', 'CRLB (analytical, theoretical)'}, ...
    'Location','best');

savefig(fig_crlb_v_ci, fullfile(fig_dir,'fig_crlb_v_bound_ci_vs_snr.fig'));
exportgraphics(fig_crlb_v_ci, fullfile(fig_dir,'fig_crlb_v_bound_ci_vs_snr.png'),'Resolution',300);

% ---- Range (eq_crlb_r) ----
fig_crlb_R_ci = figure('Name','Range CRLB bound: theory (line) vs. simulation (95% CI)', ...
    'Position',[160 160 800 520],'Color','w');

loR = rmse_R_lo;
hiR = rmse_R_hi;
validR = ~isnan(loR) & ~isnan(hiR);

fill([x(validR); flipud(x(validR))], [loR(validR); flipud(hiR(validR))], ...
    c_mle, 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
semilogy(x, rmse_R_mid, '-o', 'LineWidth', 2.0, 'Color', c_mle, 'MarkerFaceColor', c_mle);
semilogy(x, mean_crlb_R_ana, '-k', 'LineWidth', 2.2);
set(gca, 'YScale', 'log');
grid on; box on;
xlabel('SNR (dB)'); ylabel('$\sqrt{\mathrm{CRLB}_r}$ / RMSE$_r$ (m)', 'Interpreter','latex');
title('Range CRLB Bound (eq\_crlb\_r): Theoretical Line vs. Numerical 95% CI');
legend({'95% CI (bootstrap, MLE)', 'Mean RMSE (MLE, numerical)', 'CRLB (analytical, theoretical)'}, ...
    'Location','best');

savefig(fig_crlb_R_ci, fullfile(fig_dir,'fig_crlb_R_bound_ci_vs_snr.fig'));
exportgraphics(fig_crlb_R_ci, fullfile(fig_dir,'fig_crlb_R_bound_ci_vs_snr.png'),'Resolution',300);

% =====================================================================
%  ADDITION 1: MSE-identity check  (MSE == Var + Bias^2)
%  Verifies the bias-variance decomposition numerically per SNR/target and
%  prints a summary table. Any residual > 1e-9 (relative) flags a bug.
% =====================================================================
mse_R_mle = rmse_R_mle.^2;   % MSE = RMSE^2
mse_v_mle = rmse_v_mle.^2;
identity_R = var_R_mle + bias_R_mle.^2;   % should equal mse_R_mle
identity_v = var_v_mle + bias_v_mle.^2;   % should equal mse_v_mle

fprintf('\n===== MSE = Var + Bias^2 identity check (MLE) =====\n');
max_rel_err = 0;
for idx_snr = 1:length(SNR_Tx)
    for j = 1:N_tars
        for which = ["R","v"]
            if which == "R"
                mse_val = mse_R_mle(idx_snr,j); id_val = identity_R(idx_snr,j);
            else
                mse_val = mse_v_mle(idx_snr,j); id_val = identity_v(idx_snr,j);
            end
            rel = abs(mse_val - id_val) / max(abs(mse_val), eps);
            max_rel_err = max(max_rel_err, rel);
        end
    end
end
fprintf('Max relative |MSE - (Var+Bias^2)| over all SNR/targets/params = %.3e\n', max_rel_err);
if max_rel_err < 1e-9
    fprintf('Identity holds to numerical precision. OK.\n');
else
    warning('MSE decomposition identity residual exceeds 1e-9; check the estimators.');
end

% =====================================================================
%  ADDITION 2: bias-significance test + bias-with-95%-CI plot
%  For each SNR, pool the MLE residuals across targets and (a) bootstrap a
%  95% CI on the mean (bias), (b) compute the z-score z = bias/(std/sqrt(N)).
%  |z|>1.96 (or CI excluding 0) => statistically significant bias at 95%.
% =====================================================================
n_boot   = 2000;
alpha_ci = 0.05;
rng(2);  % separate reproducible stream for the bias bootstrap

[bias_v_lo, bias_v_hi, bias_v_mid, z_v] = local_bootstrap_mean_ci( ...
    all_err_v_mle_store, n_boot, alpha_ci);
[bias_R_lo, bias_R_hi, bias_R_mid, z_R] = local_bootstrap_mean_ci( ...
    all_err_R_mle_store, n_boot, alpha_ci);

sig_v = (bias_v_lo > 0) | (bias_v_hi < 0);   % CI excludes zero
sig_R = (bias_R_lo > 0) | (bias_R_hi < 0);

fprintf('\n===== Bias-significance test (MLE, pooled over targets) =====\n');
fprintf('%6s | %12s %10s %6s | %12s %10s %6s\n', ...
    'SNR dB', 'Bias_R (m)', 'z_R', 'sig?', 'Bias_v (m/s)', 'z_v', 'sig?');
for idx_snr = 1:length(SNR_Tx)
    fprintf('%6.1f | %12.3e %10.2f %6s | %12.3e %10.2f %6s\n', ...
        SNR_Tx_db(idx_snr), ...
        bias_R_mid(idx_snr), z_R(idx_snr), string(sig_R(idx_snr)), ...
        bias_v_mid(idx_snr), z_v(idx_snr), string(sig_v(idx_snr)));
end

fig_bias_ci = figure('Name','Estimator bias with 95% CI (bias-significance)', ...
    'Position',[180 180 1400 520],'Color','w');

subplot(1,2,1);
xr = SNR_Tx_db(:);
vr = ~isnan(bias_R_lo) & ~isnan(bias_R_hi);
fill([xr(vr); flipud(xr(vr))], [bias_R_lo(vr); flipud(bias_R_hi(vr))], ...
    c_mle, 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
plot(xr, bias_R_mid, '-o', 'LineWidth', 2.0, 'Color', c_mle, 'MarkerFaceColor', c_mle);
yline(0, '-k', 'LineWidth', 1.2);
grid on; box on;
xlabel('SNR (dB)'); ylabel('Range bias (m)');
title('Range Bias with 95% CI (CI excluding 0 = biased)');
legend({'95% CI (bootstrap)','Empirical bias','Zero bias'}, 'Location','best');

subplot(1,2,2);
vv = ~isnan(bias_v_lo) & ~isnan(bias_v_hi);
fill([xr(vv); flipud(xr(vv))], [bias_v_lo(vv); flipud(bias_v_hi(vv))], ...
    c_mle, 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
plot(xr, bias_v_mid, '-o', 'LineWidth', 2.0, 'Color', c_mle, 'MarkerFaceColor', c_mle);
yline(0, '-k', 'LineWidth', 1.2);
grid on; box on;
xlabel('SNR (dB)'); ylabel('Velocity bias (m/s)');
title('Velocity Bias with 95% CI (CI excluding 0 = biased)');
legend({'95% CI (bootstrap)','Empirical bias','Zero bias'}, 'Location','best');

sgtitle('Estimator Bias vs. SNR with 95% Bootstrap CI');

savefig(fig_bias_ci, fullfile(fig_dir,'fig_bias_significance_vs_snr.fig'));
exportgraphics(fig_bias_ci, fullfile(fig_dir,'fig_bias_significance_vs_snr.png'),'Resolution',300);

% =====================================================================
%  ADDITION 3: Bias^2 vs Variance vs SNR  (high-SNR floor test)
%  Separates the two MSE contributions. An unbiased/efficient estimator has
%  Bias^2 -> 0 (both terms decay); a biased estimator shows a Bias^2 FLOOR
%  that persists while Var keeps decreasing, so MSE plateaus above zero.
% =====================================================================
mean_var_R  = mean(var_R_mle, 2, 'omitnan');
mean_var_v  = mean(var_v_mle, 2, 'omitnan');
mean_bias2_R = mean(bias_R_mle.^2, 2, 'omitnan');
mean_bias2_v = mean(bias_v_mle.^2, 2, 'omitnan');
mean_mse_R  = mean(mse_R_mle, 2, 'omitnan');
mean_mse_v  = mean(mse_v_mle, 2, 'omitnan');
crlb_R_var  = mean(crlb_R_ana.^2, 2, 'omitnan');   % CRLB variance (= (sqrt CRLB)^2)
crlb_v_var  = mean(crlb_v_ana.^2, 2, 'omitnan');

c_var   = [0.00 0.45 0.74];
c_bias2 = [0.85 0.33 0.10];
c_mse   = [0.20 0.20 0.20];

fig_bv = figure('Name','Bias^2 vs Variance vs SNR (high-SNR floor test)', ...
    'Position',[200 200 1400 520],'Color','w');

subplot(1,2,1);
semilogy(SNR_Tx_db, mean_var_R,   '-o','LineWidth',1.8,'Color',c_var); hold on;
semilogy(SNR_Tx_db, mean_bias2_R, '-s','LineWidth',1.8,'Color',c_bias2);
semilogy(SNR_Tx_db, mean_mse_R,   '-^','LineWidth',2.0,'Color',c_mse);
semilogy(SNR_Tx_db, crlb_R_var,   ':k','LineWidth',1.6);
grid on; box on;
xlabel('SNR (dB)'); ylabel('Range error power (m^2)');
title('Range: Bias^2 vs Variance vs SNR');
legend({'Var','Bias^2','MSE = Var + Bias^2','CRLB variance'}, 'Location','best');

subplot(1,2,2);
semilogy(SNR_Tx_db, mean_var_v,   '-o','LineWidth',1.8,'Color',c_var); hold on;
semilogy(SNR_Tx_db, mean_bias2_v, '-s','LineWidth',1.8,'Color',c_bias2);
semilogy(SNR_Tx_db, mean_mse_v,   '-^','LineWidth',2.0,'Color',c_mse);
semilogy(SNR_Tx_db, crlb_v_var,   ':k','LineWidth',1.6);
grid on; box on;
xlabel('SNR (dB)'); ylabel('Velocity error power ((m/s)^2)');
title('Velocity: Bias^2 vs Variance vs SNR');
legend({'Var','Bias^2','MSE = Var + Bias^2','CRLB variance'}, 'Location','best');

sgtitle('Bias^2 vs. Variance Decomposition vs. SNR (Bias^2 floor = biased estimator)');

savefig(fig_bv, fullfile(fig_dir,'fig_bias2_vs_variance_vs_snr.fig'));
exportgraphics(fig_bv, fullfile(fig_dir,'fig_bias2_vs_variance_vs_snr.png'),'Resolution',300);

% =====================================================================
function [lo, hi, mid] = local_bootstrap_rmse_ci(err_store, n_boot, alpha_ci)
% Percentile bootstrap CI on the RMSE statistic, pooling trials across
% targets at each SNR point in err_store (a cell array, one Mc x N_tars
% error matrix per SNR).
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

function [lo, hi, mid, zscore] = local_bootstrap_mean_ci(err_store, n_boot, alpha_ci)
% Percentile bootstrap CI on the MEAN (bias) of the residuals, pooling
% trials across targets at each SNR point. Also returns the z-score
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
