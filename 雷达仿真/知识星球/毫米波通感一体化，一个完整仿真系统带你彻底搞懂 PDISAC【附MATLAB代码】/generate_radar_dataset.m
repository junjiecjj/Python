clear; clc; close all;

addpath(fileparts(mfilename('fullpath')));
main1_matlab_config;
G_tx_db = dataset_train_cfg.system_override.gain_tx_db;
G_rx_db = dataset_train_cfg.system_override.gain_rx_db;
G_ue_db = dataset_train_cfg.system_override.gain_ue_db;
G_tx = 10^(G_tx_db/10); G_rx = 10^(G_rx_db/10); G_ue = 10^(G_ue_db/10);

% =====================================================================
%  SIMULATION DIMENSIONS
% =====================================================================
N_ran        = dataset_train_cfg.random_scenes;
Mc           = dataset_train_cfg.monte_carlo;
N_sym_list   = dataset_train_cfg.symbols_per_block(:).';
N_sym_cases  = length(N_sym_list);

SNR_Tx_db = dataset_train_cfg.snr_db(:).';
SNR_Tx    = 10.^(SNR_Tx_db/10);

% ---------------------------------------------------------------------
%  SMOKE-TEST MODE  (run with env var PDISAC_SMOKE=1)
%  Shrinks the sweep and writes dataset_smoke.db for a quick code check.
%  producing a tiny database only for pipeline execution checks.
% ---------------------------------------------------------------------
smoke_mode = ~isempty(getenv('PDISAC_SMOKE'));
if smoke_mode
    N_ran       = dataset_smoke_cfg.random_scenes;
    Mc          = dataset_smoke_cfg.monte_carlo;
    N_sym_list  = dataset_smoke_cfg.symbols_per_block(:).';
    N_sym_cases = length(N_sym_list);
    SNR_Tx_db   = dataset_smoke_cfg.snr_db(:).';
    SNR_Tx      = 10.^(SNR_Tx_db/10);
    G_tx_db = dataset_smoke_cfg.system_override.gain_tx_db;
    G_rx_db = dataset_smoke_cfg.system_override.gain_rx_db;
    G_ue_db = dataset_smoke_cfg.system_override.gain_ue_db;
    G_tx = 10^(G_tx_db/10); G_rx = 10^(G_rx_db/10); G_ue = 10^(G_ue_db/10);
    fprintf('*** SMOKE-TEST MODE: N_ran=%d, N_sym=[%s], SNR=[%s] dB ***\n', ...
        N_ran, num2str(N_sym_list), num2str(SNR_Tx_db));
end

% Region of interest  [min max] per axis  (x ; y ; z)
% 5 fixed targets + 1 UE = 6 total
N_tars_fixed = dataset_train_cfg.fixed_targets;
N_tars_total = N_tars_fixed + 1;

% =====================================================================
%  RESULTS BUFFER SETUP (SCALAR TABLE)
% =====================================================================
% KEY ALIGNMENT with the DB scalar_results table:
%   Each row is indexed by (Trial_ID, RAN_Index, N_sym_per_block, SNR_dB, MC_Index).
%   All N_tars_total rows sharing the same Trial_ID correspond to one RD map,
%   retrievable from matrix_data via the same Trial_ID foreign key.

col_names_scalar = { ...
    'Trial_ID', ...         % Relational linking key → joins scalar_results ↔ matrix_data
    'RAN_Index', ...
    'N_sym_per_block', ...
    'Tar_X','Tar_Y','Tar_Z', ...
    'Tar_Vx','Tar_Vy','Tar_Vz', ...
    'Tar_RCS', ...
    'SNR_dB', ...
    'True_Range_m',    'Est_Range_cfar_m', ...
    'True_Vel_mps',    'Est_Vel_cfar_mps', ...
    'CRLB_Range_ana_m','CRLB_Range_sim_m', ...
    'CRLB_Vel_ana_mps','CRLB_Vel_sim_mps', ...
    'MC_Index'};

n_rows_full = N_ran * N_sym_cases * length(SNR_Tx) * Mc * N_tars_total;
results_buf = nan(n_rows_full, numel(col_names_scalar));
row_ptr     = 1;

% =====================================================================
%  RESULTS BUFFER SETUP (MATRIX TABLE)
% =====================================================================
% One row per (idx_ran × idx_nsym × idx_snr × idx_mc) trial.
% RD maps stored as lossless float32 GZIP+base64 strings, keyed by Trial_ID.

n_trials_total = N_ran * N_sym_cases * length(SNR_Tx) * Mc;
trial_id       = 0;

matrix_buf_Trial_ID    = (1:n_trials_total)';
matrix_buf_RD          = repmat("", n_trials_total, 1);
matrix_buf_RD_no_noise = repmat("", n_trials_total, 1);

% =====================================================================
%  METRICS  (N_ran x N_sym_cases x N_snr x N_tars_total)
% =====================================================================
rmse_R     = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);
rmse_v     = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);
bias_R     = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);
bias_v     = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);
crlb_R_ana = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);
crlb_v_ana = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);
crlb_R_sim = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);
crlb_v_sim = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);
pd_per     = nan(N_ran, N_sym_cases, length(SNR_Tx), N_tars_total);

% =====================================================================
%  LOOP 1 – random scene realisation
%           UE, targets, and scatterers drawn ONCE and held fixed
%           across all N_sym / SNR / Mc inner loops
% =====================================================================
for idx_ran = 1:N_ran

    fprintf('\n========== Scene realisation %d / %d ==========\n', idx_ran, N_ran);

    % ------------------------------------------------------------------
    %  Draw UE
    % ------------------------------------------------------------------
    N_ue        = 1;
    UE_position = zeros(3, 1);
    for dim = 1:3
        lo = Region_of_interest(dim, 1);
        hi = Region_of_interest(dim, 2);
        if lo == hi
            UE_position(dim) = lo;
        else
            UE_position(dim) = lo + (hi - lo) * randn();
        end
    end
    UE_vel    = VEL_RANGE_UE * (2*randn(3,1) - 1);
    UE_vel(3) = 0;
    UE_rcs    = 10^( (10*log10(10) + (10*log10(20)-10*log10(10))*randn()) / 10 );

    UE_motion = phased.Platform( ...
        'InitialPosition', UE_position, ...
        'Velocity',        UE_vel);

    % ------------------------------------------------------------------
    %  Draw targets
    % ------------------------------------------------------------------
    N_tars        = N_tars_fixed;
    Tars_position = zeros(3, N_tars);
    for dim = 1:3
        lo = Region_of_interest(dim, 1);
        hi = Region_of_interest(dim, 2);
        if lo == hi
            Tars_position(dim, :) = lo;
        else
            Tars_position(dim, :) = lo + (hi - lo) * randn(1, N_tars);
        end
    end
    Tars_vel      = VEL_RANGE_TAR * (2*randn(3, N_tars) - 1);
    Tars_vel(3,:) = 0;
    Tars_rcs      = 10.^( (10*log10(10) + ...
                           (10*log10(20)-10*log10(10))*randn(1, N_tars)) / 10 );

    % Append UE as the last target  (index N_tars_total = 6)
    N_tars        = N_tars + 1;
    Tars_position = [Tars_position, UE_position];
    Tars_vel      = [Tars_vel,      UE_vel];
    Tars_rcs      = [Tars_rcs,      UE_rcs];

    Tars_motion = phased.Platform( ...
        'InitialPosition', Tars_position, ...
        'Velocity',        Tars_vel);

    % ------------------------------------------------------------------
    %  Draw scatterers
    % ------------------------------------------------------------------
    N_scats        = dataset_train_cfg.scatterers;
    Scats_position = func_generate_static_scatterers(N_scats, Region_of_interest);
    Scats_vel      = zeros(size(Scats_position));
    Scats_rcs      = 10.^( (10*log10(5) + ...
                            (10*log10(10)-10*log10(5))*randn(1, N_scats)) / 10 );
    Scats_motion   = phased.Platform( ...
        'InitialPosition', Scats_position, ...
        'Velocity',        Scats_vel);

    N_ref_tars = PDISAC_cfg.scene.communication_reflectors;

    % Ground-truth range & radial velocity (fixed for this realisation)
    truth_range = sqrt(sum((Tars_position - Tx_position).^2, 1)).';
    unit_vec    = (Tars_position - Tx_position) ./ ...
                  (sqrt(sum((Tars_position - Tx_position).^2, 1)) + eps);
    truth_vel   = sum(Tars_vel .* unit_vec, 1).';   % (N_tars × 1)

    % ==================================================================
    %  LOOP 2 – N_sym_per_block sweep
    % ==================================================================
    for idx_nsym = 1:N_sym_cases

        N_sym_per_block = N_sym_list(idx_nsym);

        % Waveform parameters that depend on N_sym_per_block
        L_slot_per_sym = N_chip / (2 * N_sym_per_block);
        N_slot         = N_chip / L_slot_per_sym;

        fprintf('\n  ----- N_sym_per_block = %d -----\n', N_sym_per_block);

        % ================================================================
        %  LOOP 3 – SNR sweep
        % ================================================================
        for idx_snr = 1:length(SNR_Tx)

            snr_tx = SNR_Tx(idx_snr);
            P_tx   = Noise_power_sen * snr_tx;

            % Per-Mc error accumulators  (reset for each SNR point)
            all_err_R_cfar = nan(Mc, N_tars);
            all_err_v_cfar = nan(Mc, N_tars);
            all_err_R_mle  = nan(Mc, N_tars);
            all_err_v_mle  = nan(Mc, N_tars);
            est_R_cfar     = nan(Mc, N_tars);
            est_v_cfar     = nan(Mc, N_tars);
            det_count      = zeros(1, N_tars);

            % ============================================================
            %  LOOP 4 – Monte-Carlo  (signal noise only; scene is fixed)
            % ============================================================
            for idx_mc = 1:Mc

                % --------------------------------------------------------
                %  Signal pipeline
                % --------------------------------------------------------
                main3_signal_channel_model;
                
                main4_sensing_process;
                % --------------------------------------------------------
                %  Increment Trial_ID and compress RD maps
                %  (lossless float32 GZIP+base64, keyed for DB join)
                % --------------------------------------------------------
                trial_id = trial_id + 1;
                matrix_buf_RD(trial_id)          = compressMatrix(RD_map_shifted);
                matrix_buf_RD_no_noise(trial_id) = compressMatrix(RD_map_no_noise_shifted);

                % --------------------------------------------------------
                %  Range / velocity axes
                % --------------------------------------------------------
                range_res  = c / (2 * B);
                range_axis = (0 : N_chip-1) * range_res;

                PRF      = 1 / T_pmcw;
                fd_axis  = (-N_block/2 : N_block/2-1) * (PRF / N_block);
                vel_axis = -(Lambda / 2) * fd_axis;

                % --------------------------------------------------------
                %  CA-CFAR
                % --------------------------------------------------------
                N_guard_range   = PDISAC_cfg.sensing.cfar.guard_range;
                N_guard_doppler = PDISAC_cfg.sensing.cfar.guard_doppler;
                N_train_range   = PDISAC_cfg.sensing.cfar.training_range;
                N_train_doppler = PDISAC_cfg.sensing.cfar.training_doppler;
                P_fa = PDISAC_cfg.sensing.cfar.false_alarm_probability;
                peak_select = PDISAC_cfg.sensing.cfar.peak_select;

                [threshold_map, noise_power_avg_map, detected_positions, RD_power_map] = ...
                    func_ca_cfar_adaptive_threshold( ...
                        RD_map_shifted, ...
                        N_guard_range, N_guard_doppler, ...
                        N_train_range, N_train_doppler, ...
                        P_fa, peak_select);

                K_det = size(detected_positions, 1);

                % --------------------------------------------------------
                %  Sinc interpolation on detections
                % --------------------------------------------------------
                if K_det > 0
                    [det_range, det_vel] = func_sinc_interpolation( ...
                        RD_map_shifted, detected_positions, range_axis, vel_axis);
                end

                % --------------------------------------------------------
                %  Nearest-neighbour target association
                % --------------------------------------------------------
                assoc          = nan(N_tars, 1);
                err_R_vec_cfar = nan(N_tars, 1);
                err_v_vec_cfar = nan(N_tars, 1);

                roi_range  = Region_of_interest(1, :);
                roi_vel    = Region_of_interest(2, :);

                if K_det > 0
                    range_span = max(diff(roi_range), eps);
                    vel_span   = max(diff(roi_vel),   eps);
                    for j = 1:N_tars
                        dist2 = ((det_range - truth_range(j)) / range_span).^2 + ...
                                ((det_vel   - truth_vel(j)  ) / vel_span  ).^2;
                        [~, assoc(j)] = min(dist2);
                        err_R_vec_cfar(j) = det_range(assoc(j)) - truth_range(j);
                        err_v_vec_cfar(j) = det_vel(assoc(j))   - truth_vel(j);
                        det_count(j)      = det_count(j) + 1;
                    end
                    est_R_cfar(idx_mc, :) = det_range(assoc);
                    est_v_cfar(idx_mc, :) = det_vel(assoc);
                end

                all_err_R_cfar(idx_mc, :) = err_R_vec_cfar;
                all_err_v_cfar(idx_mc, :) = err_v_vec_cfar;

                % --------------------------------------------------------
                %  MLE estimator  (used as simulated CRLB proxy)
                % --------------------------------------------------------
                err_R_vec_mle = nan(N_tars, 1);
                err_v_vec_mle = nan(N_tars, 1);

                [current_est_R_mle, current_est_v_mle] = func_mle_estimation( ...
                    Data_Rx, Data_Tx, N_tars, ...
                    Tars_position, Tars_rcs, Tars_vel, ...
                    Tx_position, N_tx_ant, Rx_position, N_rx_ant, ...
                    P_tx, G_tx, G_rx, Lambda, fc, Noise_power_sen, ...
                    T_chip, N_chip, N_block);

                for j = 1:N_tars
                    err_R_vec_mle(j) = current_est_R_mle(j) - truth_range(j);
                    err_v_vec_mle(j) = current_est_v_mle(j) - truth_vel(j);
                end

                all_err_R_mle(idx_mc, :) = err_R_vec_mle;
                all_err_v_mle(idx_mc, :) = err_v_vec_mle;

                % --------------------------------------------------------
                %  Write scalar results row  (CRLB columns back-filled below)
                % --------------------------------------------------------
                for j = 1:N_tars
                    results_buf(row_ptr, :) = [ ...
                        trial_id, ...
                        idx_ran, ...
                        N_sym_per_block, ...
                        Tars_position(1,j), Tars_position(2,j), Tars_position(3,j), ...
                        Tars_vel(1,j),      Tars_vel(2,j),      Tars_vel(3,j),      ...
                        Tars_rcs(j),        ...
                        SNR_Tx_db(idx_snr), ...
                        truth_range(j),     est_R_cfar(idx_mc, j), ...
                        truth_vel(j),       est_v_cfar(idx_mc, j), ...
                        nan, nan, nan, nan, ...   % CRLB — filled below
                        idx_mc];
                    row_ptr = row_ptr + 1;
                end

            end  % idx_mc  ---------------------------------------------

            % ------------------------------------------------------------
            %  Aggregate metrics over MC
            % ------------------------------------------------------------
            rmse_R(idx_ran, idx_nsym, idx_snr, :) = sqrt(mean(all_err_R_cfar.^2, 1, 'omitnan'));
            rmse_v(idx_ran, idx_nsym, idx_snr, :) = sqrt(mean(all_err_v_cfar.^2, 1, 'omitnan'));
            bias_R(idx_ran, idx_nsym, idx_snr, :) = mean(all_err_R_cfar, 1, 'omitnan');
            bias_v(idx_ran, idx_nsym, idx_snr, :) = mean(all_err_v_cfar, 1, 'omitnan');
            pd_per(idx_ran, idx_nsym, idx_snr, :) = det_count / Mc;

            % Simulated CRLB from MLE RMSE
            crlb_R_sim(idx_ran, idx_nsym, idx_snr, :) = sqrt(mean(all_err_R_mle.^2, 1, 'omitnan'));
            crlb_v_sim(idx_ran, idx_nsym, idx_snr, :) = sqrt(mean(all_err_v_mle.^2, 1, 'omitnan'));

            % Analytical CRLB — geometry fixed for this idx_ran
            [cur_crlb_r_ana, cur_crlb_v_ana] = func_ana_crlb_rv( ...
                N_tars, Tars_position, Tars_rcs, Tars_vel, ...
                Tx_position, N_tx_ant, Rx_position, N_rx_ant, ...
                P_tx, G_tx, G_rx, Lambda, fc, Noise_power_sen, ...
                T_chip, N_chip, N_block);

            crlb_R_ana(idx_ran, idx_nsym, idx_snr, :) = sqrt(cur_crlb_r_ana);
            crlb_v_ana(idx_ran, idx_nsym, idx_snr, :) = sqrt(cur_crlb_v_ana);

            % ------------------------------------------------------------
            %  Back-fill CRLB into rows written during this block
            % ------------------------------------------------------------
            n_rows_block    = Mc * N_tars;
            rows_this_block = (row_ptr - n_rows_block) : (row_ptr - 1);

            for mc_i = 1:Mc
                for j = 1:N_tars
                    r = rows_this_block(1) + (mc_i-1)*N_tars + (j-1);
                    results_buf(r, 16) = crlb_R_ana(idx_ran, idx_nsym, idx_snr, j);  % CRLB_Range_ana_m
                    results_buf(r, 17) = crlb_R_sim(idx_ran, idx_nsym, idx_snr, j);  % CRLB_Range_sim_m
                    results_buf(r, 18) = crlb_v_ana(idx_ran, idx_nsym, idx_snr, j);  % CRLB_Vel_ana_mps
                    results_buf(r, 19) = crlb_v_sim(idx_ran, idx_nsym, idx_snr, j);  % CRLB_Vel_sim_mps
                end
            end

            fprintf('RAN=%2d | Nsym=%3d | SNR=%+3d dB | RMSE_R=%.3f m  CRLB_R_ana=%.3f m  CRLB_R_sim=%.3f m | RMSE_v=%.3f m/s  CRLB_v_ana=%.3f m/s  CRLB_v_sim=%.3f m/s | P_d=%.2f\n', ...
                idx_ran, N_sym_per_block, SNR_Tx_db(idx_snr), ...
                mean(rmse_R(idx_ran,idx_nsym,idx_snr,:), 'omitnan'), ...
                mean(crlb_R_ana(idx_ran,idx_nsym,idx_snr,:), 'omitnan'), ...
                mean(crlb_R_sim(idx_ran,idx_nsym,idx_snr,:), 'omitnan'), ...
                mean(rmse_v(idx_ran,idx_nsym,idx_snr,:), 'omitnan'), ...
                mean(crlb_v_ana(idx_ran,idx_nsym,idx_snr,:), 'omitnan'), ...
                mean(crlb_v_sim(idx_ran,idx_nsym,idx_snr,:), 'omitnan'), ...
                mean(pd_per(idx_ran,idx_nsym,idx_snr,:), 'omitnan'));

        end  % idx_snr  -----------------------------------------------
    end  % idx_nsym  ---------------------------------------------------
end  % idx_ran  ---------------------------------------------------------


% =====================================================================
%  SAVE COMPRESSED DATA TO SQLITE DATABASE
%
%  Schema (two tables, joined on Trial_ID):
%
%  Table 1 — scalar_results
%    One row per (Trial_ID × target).  N_tars_total rows share each Trial_ID.
%    Columns: Trial_ID (FK) + all 19 scalar metrics (see col_names_scalar).
%
%  Table 2 — matrix_data
%    One row per Trial_ID.  RD maps stored as lossless float32 GZIP+base64.
%    Decompress: base64decode → gunzip → typecast('single') → reshape(N_chip, N_block).
%    Columns: Trial_ID (PK), RD_map_compressed, RD_map_no_noise_compressed.
%
%  Join example (SQL):
%    SELECT s.*, m.RD_map_compressed
%    FROM scalar_results s
%    JOIN matrix_data m ON s.Trial_ID = m.Trial_ID
%    WHERE s.SNR_dB = 10 AND s.N_sym_per_block = 32;
% =====================================================================
fprintf('\nInitializing Relational SQLite Database Export...\n');
db_filename = fullfile(PDISAC_repo_root, dataset_train_cfg.output_db);
if smoke_mode
    db_filename = fullfile(PDISAC_repo_root, dataset_smoke_cfg.output_db);
end

if isfile(db_filename)
    delete(db_filename);   % Wipe existing file to avoid structure lockouts
end

conn = sqlite(db_filename, 'create');

% -- Table 1: Scalar Evaluation Metrics --
scalar_table = array2table(results_buf, 'VariableNames', col_names_scalar);
sqlwrite(conn, 'scalar_results', scalar_table);
fprintf('-> Table [scalar_results] successfully committed.\n');

% -- Table 2: Normalised Matrix Storage (RD maps, keyed by Trial_ID) --
matrix_table = table( ...
    matrix_buf_Trial_ID, ...
    matrix_buf_RD, ...
    matrix_buf_RD_no_noise, ...
    'VariableNames', {'Trial_ID', 'RD_map_compressed', 'RD_map_no_noise_compressed'});
sqlwrite(conn, 'matrix_data', matrix_table);
fprintf('-> Table [matrix_data] successfully committed.\n');

close(conn);
fprintf('Database generation process complete. Binary written to: %s\n', db_filename);

% =====================================================================
%  PLOTS  (averaged over N_ran and targets, one curve per N_sym)
% =====================================================================
fig_rmse_nsym = figure('Name','Sensing: RMSE vs SNR by N_sym','Position',[60 60 1400 560],'Color','w');
cmap = lines(N_sym_cases);

subplot(1,2,1); hold on; grid on; box on;
for i = 1:N_sym_cases
    % mean over idx_ran (dim1) and N_tars (dim4)
    curve = squeeze(mean(mean(rmse_R(:,i,:,:), 1, 'omitnan'), 4, 'omitnan'));
    semilogy(SNR_Tx_db, curve(:)' / 4, '-o', 'Color', cmap(i,:), 'LineWidth', 1.4, ...
        'MarkerSize', 5, 'DisplayName', sprintf('RMSE  N_{sym}=%d', N_sym_list(i)));
end
crlb_R_ana_mean = squeeze(mean(mean(crlb_R_ana(:,end,:,:), 1, 'omitnan'), 4, 'omitnan'));
crlb_R_sim_mean = squeeze(mean(mean(crlb_R_sim(:,end,:,:), 1, 'omitnan'), 4, 'omitnan'));
semilogy(SNR_Tx_db, crlb_R_ana_mean(:)', 'k-',  'LineWidth', 2.2, 'DisplayName', 'CRLB (analytical)');
semilogy(SNR_Tx_db, crlb_R_sim_mean(:)', 'r--', 'LineWidth', 2.2, 'DisplayName', 'CRLB (sim / MLE)');
xlabel('SNR (dB)', 'FontSize', 12); ylabel('RMSE  [m]', 'FontSize', 12);
title('Range estimation', 'FontSize', 13); legend('Location','best','FontSize',9);

subplot(1,2,2); hold on; grid on; box on;
for i = 1:N_sym_cases
    curve = squeeze(mean(mean(rmse_v(:,i,:,:), 1, 'omitnan'), 4, 'omitnan'));
    semilogy(SNR_Tx_db, curve(:)', '-o', 'Color', cmap(i,:), 'LineWidth', 1.4, ...
        'MarkerSize', 5, 'DisplayName', sprintf('RMSE  N_{sym}=%d', N_sym_list(i)));
end
crlb_v_ana_mean = squeeze(mean(mean(crlb_v_ana(:,end,:,:), 1, 'omitnan'), 4, 'omitnan'));
crlb_v_sim_mean = squeeze(mean(mean(crlb_v_sim(:,end,:,:), 1, 'omitnan'), 4, 'omitnan'));
semilogy(SNR_Tx_db, crlb_v_ana_mean(:)', 'k-',  'LineWidth', 2.2, 'DisplayName', 'CRLB (analytical)');
semilogy(SNR_Tx_db, crlb_v_sim_mean(:)', 'r--', 'LineWidth', 2.2, 'DisplayName', 'CRLB (sim / MLE)');
xlabel('SNR (dB)', 'FontSize', 12); ylabel('RMSE  [m/s]', 'FontSize', 12);
title('Velocity estimation', 'FontSize', 13); legend('Location','best','FontSize',9);

sgtitle('RMSE vs SNR  —  CFAR estimator & CRLB bounds', 'FontSize', 14, 'FontWeight', 'bold');


% ---- Export figure (.fig + .png) into fig_exported ----
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
savefig(fig_rmse_nsym, fullfile(fig_dir,'fig_rmse_vs_snr_by_nsym.fig'));
exportgraphics(fig_rmse_nsym, fullfile(fig_dir,'fig_rmse_vs_snr_by_nsym.png'), 'Resolution',300);


% =====================================================================
%  LOCAL FUNCTION — Lossless Float32 GZIP + Base64 Compression
%
%  Input  : M  — complex matrix (any size, e.g. N_chip × N_block)
%  Output : base64Str — scalar string, suitable for SQLite TEXT column
%
%  Decompress (MATLAB):
%              1j * reshape(raw(nEl+1:end), size(M));
%
%  Decompress (Python):
% =====================================================================
function base64Str = compressMatrix(M)
    reBytes = typecast(single(real(M(:))), 'uint8');
    imBytes = typecast(single(imag(M(:))), 'uint8');
    rawBytes = [reBytes; imBytes];

    baos = java.io.ByteArrayOutputStream();
    gzos = java.util.zip.GZIPOutputStream(baos);
    gzos.write(rawBytes);
    gzos.close();

    base64Str = string(matlab.net.base64encode(typecast(baos.toByteArray(), 'uint8')));
end


function M = decompressMatrix(base64Str, numRows, numCols)
    % 1. Decode Base64 string back into zip-byte array
    compressedBytes = matlab.net.base64decode(base64Str);
    
    % 2. Inflate stream using Java GZIPInputStream
    bais = java.io.ByteArrayInputStream(compressedBytes);
    gzis = java.util.zip.GZIPInputStream(bais);
    baos = java.io.ByteArrayOutputStream();
    
    buffer = javaArray('byte', 4096);
    while true
        bytesRead = gzis.read(buffer, 0, 4096);
        if bytesRead == -1, break; end
        baos.write(buffer, 0, bytesRead);
    end
    gzis.close();
    rawBytes = typecast(baos.toByteArray(), 'uint8');
    
    % 3. Extract Real and Imaginary segments (4 bytes per single value)
    totalElements = numRows * numCols;
    bytesPerPart  = totalElements * 4; 
    
    re = typecast(rawBytes(1:bytesPerPart), 'single');
    im = typecast(rawBytes(bytesPerPart+1:end), 'single');
    
    % 4. Reshape back into original dimensions
    M = reshape(complex(re, im), numRows, numCols);
end
