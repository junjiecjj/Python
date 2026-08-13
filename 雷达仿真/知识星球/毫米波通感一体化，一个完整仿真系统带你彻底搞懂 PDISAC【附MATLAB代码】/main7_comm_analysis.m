% COMM_ANALYSIS_B  (variant of Group 7: communication analysis)
% -------------------------------------------------------------------------
% Same Monte-Carlo BER/throughput sweep as main7_comm_analysis.m (same
% pipeline calls: main3_signal_channel_model + main6c_comm_process, same
% BER/net-throughput computations, func_ana_ber, func_ana_net,
% func_ana_ber_closeform all UNCHANGED).
%
% CHANGE vs. main7: the capacity computation is replaced to match the
% "Ensemble-Average Capacity Analysis" of 5_7_results_comm.tex:
%     C_seq^{(i)}   = sum_{k=1}^{N_bit_prbs} log2(1+gamma_{k,i})         (eq_cap_seq)
%     Cbar_seq      = (1/N_prbs) sum_i C_seq^{(i)}                       (eq_cap_seq_avg)
%     Cbar_net      = Cbar_seq / T_prbs                                  (eq_cap_net)
% where gamma_{k,i} is the SAME despread detection SINR used for the BER
% analysis (eq_sinr_k_i_b) (not the old per-chip, despreading-gain-free
% SINR that the previous func_ana_cap used).
%
% Both a "Numerical" and a "Theoretical" variant are computed, mirroring the
% BER Numerical/Theoretical split already used in this file:
%   - Numerical  (func_cap_seq):            Cbar_seq/Cbar_net evaluated from
%     the TRUE geometry-realized channel of THIS Monte-Carlo trial (same
%     per-realization philosophy as func_ana_ber; no parametric interference
%     distribution).
%   - Theoretical (func_ana_cap_closeform): Cbar_seq/Cbar_net obtained by
%     taking E[log2(1+gamma)] in closed form over the Gamma-distributed
%     aggregate NLOS interference (same P_los/Omega geometry setup as
%     func_ana_ber_closeform, but a direct expectation instead of the BER's
%     change-of-variables/Jacobian construction).
%
% Also adds two figures after the sweep: Cbar_seq vs. SNR and Cbar_net vs.
% SNR, both per N_bit_prbs allocation, numerical (markers) + theoretical
% (dashed), in the same style as the existing BER/throughput plots.
%
% Results are written to a SEPARATE csv (statistics_communication.csv) so
% the original statistics_communication.csv / main8 STEP 6a plots are not
% disturbed.
% NOTE: long Monte-Carlo sweep; run standalone.

% Run the pipeline headless during the sweep: main3/main6 are called every
% iteration; the main6 BPSK constellation is a slow/blocking System object.
NO_SIGNAL_PLOT = true;  NO_BPSK_PLOT = true;
NO_RADAR_PLOT  = true;  NO_CFAR_PLOT = true;

main1_matlab_config;

% =====================================================================
%  Monte-Carlo BER / throughput / capacity sweep (deterministic-per-path channel)
% =====================================================================
stat_cfg = communication_statistics_cfg;
out_csv  = 'exported_statistics/statistics_communication.csv';

G_tx = 10^(stat_cfg.system_override.gain_tx_db/10);
G_rx = 10^(stat_cfg.system_override.gain_rx_db/10);
G_ue = 10^(stat_cfg.system_override.gain_ue_db/10);
N_ran       = stat_cfg.random_scenes;
Mc          = stat_cfg.monte_carlo;
N_sym_list  = stat_cfg.symbols_per_block(:).';
N_sym_cases = numel(N_sym_list);
SNR_Tx_db   = stat_cfg.snr_db(:).';
SNR_Tx      = 10.^(SNR_Tx_db / 10);

% ---- results buffer ----
n_rows_total = N_ran * N_sym_cases * numel(SNR_Tx) * Mc;
col_names = { 'RAN_Index','N_sym_per_block','UE_X','UE_Y','UE_Z', ...
    'UE_Vx','UE_Vy','UE_Vz','SNR_dB', ...
    'Numerical_BER_Est_Tau_Est_h','Numerical_BER_Est_Tau_Perf_h', ...
    'Numerical_BER_Perf_Tau_Est_h','Numerical_BER_Perf_Tau_Perf_h', ...
    'Theoretical_BER', 'Numerical_Net','Theoretical_Net', ...
    'Cap_Seq_Numerical','Cap_Net_Numerical', ...
    'Cap_Seq_Theoretical','Cap_Net_Theoretical','MC_Index'};
results_buf = nan(n_rows_total, numel(col_names));
row_ptr = 1;

% Aggregate arrays kept for the per-SNR console summary and the capacity
% figures [SNR x Mc x N_ran x N_sym]
ber_numerical_all_v4     = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);
ber_theorectical_all     = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % func_ana_ber_closeform
net_numerical_all        = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);
net_theorectical_all     = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);
cap_seq_numerical_all    = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % (eq_cap_seq_avg), Numerical
cap_net_numerical_all    = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % (eq_cap_net),     Numerical
cap_seq_theorectical_all = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % (eq_cap_seq_avg), Theoretical
cap_net_theorectical_all = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % (eq_cap_net),     Theoretical

% ---- LOOP 1: random UE realisation ----
for idx_ran = 1:N_ran
    fprintf('\n========== UE Realisation %d / %d ==========\n', idx_ran, N_ran);

    N_ue        = 1;
    UE_position = rand_pos_in_roi(Region_of_interest, N_ue);
    UE_vel      = VEL_RANGE_UE * (2*randn(3, N_ue) - 1);  UE_vel(3) = 0;
    UE_rcs      = 10.^( (10*log10(10) + (10*log10(20)-10*log10(10))*randn(1)) / 10 );

    Rel_positions = UE_position - Tx_position;
    [~,~,Range_Tx_UE] = cart2sph(Rel_positions(1),Rel_positions(2),Rel_positions(3));
    Radial_UE_vel = dot(UE_vel, Rel_positions / norm(Rel_positions));

    % ---- LOOP 2: N_sym_per_block ----
    for idx_nsym = 1:N_sym_cases
        N_sym_per_block = N_sym_list(idx_nsym);
        L_slot_per_sym  = N_chip / (2 * N_sym_per_block);
        N_slot          = N_chip / L_slot_per_sym;
        R_raw           = N_sym_per_block / T_pmcw;
        fprintf('\n  ----- N_sym_per_block = %d -----\n', N_sym_per_block);

        % ---- LOOP 3: SNR ----
        for idx_snr = 1:numel(SNR_Tx)
            P_tx = Noise_power_com * SNR_Tx(idx_snr);

            % ---- LOOP 4: Monte-Carlo (re-draw scene) ----
            for idx_mc = 1:Mc
                % re-draw moving targets, then append UE as last target
                N_tars        = 5;
                Tars_position = rand_pos_in_roi(Region_of_interest, N_tars);
                Tars_vel      = VEL_RANGE_TAR * (2*randn(3, N_tars) - 1);  Tars_vel(3,:) = 0;
                Tars_rcs      = 10.^( (10*log10(10) + (10*log10(20)-10*log10(10))*randn(1, N_tars)) / 10 );
                N_tars        = N_tars + 1;
                Tars_position = [Tars_position, UE_position];
                Tars_vel      = [Tars_vel,      UE_vel];
                Tars_rcs      = [Tars_rcs,      UE_rcs];
                Tars_motion   = phased.Platform('InitialPosition',Tars_position,'Velocity',Tars_vel);
                UE_motion     = phased.Platform('InitialPosition',UE_position,'Velocity',UE_vel);

                % re-draw stationary scatterers
                N_scats        = 100;
                Scats_position = func_generate_static_scatterers(N_scats, Region_of_interest);
                Scats_vel      = zeros(size(Scats_position));
                Scats_rcs      = 10.^( (10*log10(5) + (10*log10(10)-10*log10(5))*randn(1, N_scats)) / 10 );
                Scats_motion   = phased.Platform('InitialPosition',Scats_position,'Velocity',Scats_vel);

                N_ref_tars = 5;   % use N_ref_tars closest reflectors to the UE

                % --- signal pipeline ---
                main3_signal_channel_model;
                main6_comm_process;

                % --- numerical BER (4 CSI/delay conditions) ---
                ber_v1 = mean(Data_UE_est_tau_est_h_bin(:)  ~= Data_Tx_bin(:));
                ber_v2 = mean(Data_UE_est_tau_perf_h_bin(:) ~= Data_Tx_bin(:));
                ber_v3 = mean(Data_UE_perf_tau_est_h_bin(:) ~= Data_Tx_bin(:));
                ber_v4 = mean(Data_UE_perf_tau_perf_h_bin(:)~= Data_Tx_bin(:));
                ber_numerical_all_v4(idx_snr, idx_mc, idx_ran, idx_nsym) = ber_v4;

                % --- semi-analytical / closed-form BER (UNCHANGED vs. main7) ---
                ber_theo = func_ana_ber_closeform(P_tx, G_tx, G_ue, L_slot_per_sym, N_ref_tars, ...
                                                Noise_power_com, UE_position, Lambda, ...
                                                N_tars-1, N_scats, Region_of_interest);
                ber_theorectical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = ber_theo;

                % --- throughput (UNCHANGED vs. main7) ---
                r_c     = 1;
                net_num = R_raw * (1 - ber_v4);
                net_numerical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = net_num;
                net_theo = func_ana_net(H_com_los_all, H_com_nlos_all, C_w_time, ...
                    Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block, R_raw, r_c);
                net_theorectical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = net_theo;

                % --- ensemble-average capacity, (eq_cap_seq)-(eq_cap_net) ---
                % Numerical: from the TRUE realized channel of this MC trial.
                [Cseq_num, Cnet_num] = func_cap_seq(H_com_los_all, H_com_nlos_all, C_w_time, ...
                    Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block, T_pmcw);
                cap_seq_numerical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = Cseq_num;
                cap_net_numerical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = Cnet_num;

                % Theoretical: closed-form E[log2(1+gamma)] over Gamma-distributed P_nlos.
                [Cseq_theo, Cnet_theo] = func_ana_cap_closeform(P_tx, G_tx, G_ue, L_slot_per_sym, ...
                    N_sym_per_block, N_ref_tars, Noise_power_com, UE_position, Lambda, ...
                    N_tars-1, N_scats, Region_of_interest, T_pmcw);
                cap_seq_theorectical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = Cseq_theo;
                cap_net_theorectical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = Cnet_theo;

                % --- store row ---
                results_buf(row_ptr, :) = [ idx_ran, N_sym_per_block, ...
                    UE_position(1), UE_position(2), UE_position(3), ...
                    UE_vel(1), UE_vel(2), UE_vel(3), SNR_Tx_db(idx_snr), ...
                    ber_v1, ber_v2, ber_v3, ber_v4, ber_theo, ...
                    net_num, net_theo, ...
                    Cseq_num, Cnet_num, Cseq_theo, Cnet_theo, idx_mc ];
                row_ptr = row_ptr + 1;
            end % idx_mc

            fprintf(['    Nsym=%3d | r_ue=%5.1f m | v_ue=%+5.1f m/s | SNR=%+3d dB | ' ...
                     'BER(num)=%.4e | BER(theo)=%.4e | ' ...
                     'Net(num)=%.4e | Net(theo)=%.4e | ' ...
                     'Cseq(num)=%.4e | Cseq(theo)=%.4e | ' ...
                     'Cnet(num)=%.4e | Cnet(theo)=%.4e\n'], ...
                N_sym_per_block, Range_Tx_UE, Radial_UE_vel, SNR_Tx_db(idx_snr), ...
                mean(ber_numerical_all_v4(idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(ber_theorectical_all(idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(net_numerical_all   (idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(net_theorectical_all(idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(cap_seq_numerical_all   (idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(cap_seq_theorectical_all(idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(cap_net_numerical_all   (idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(cap_net_theorectical_all(idx_snr,:,idx_ran,idx_nsym), 'omitnan'));
        end % idx_snr
    end % idx_nsym
end % idx_ran

% ---- write CSV once, after the full sweep (append if the file exists) ----
results_table = array2table(results_buf, 'VariableNames', col_names);
if isfile(out_csv)
    writetable(results_table, out_csv, 'WriteMode', 'append');
else
    writetable(results_table, out_csv);
end
fprintf('\nResults saved to %s  (%d rows x %d cols)\n', ...
    out_csv, height(results_table), width(results_table));

% =====================================================================
%  Capacity figures: Cbar_seq [bits/sequence] and Cbar_net [bits/s] vs. SNR,
%  per N_bit_prbs allocation, Numerical (markers) + Theoretical (dashed).
%  (eq_cap_seq_avg), (eq_cap_net) (5_7_results_comm.tex).
% =====================================================================
script_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(script_dir, 'fig_exported');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

% Okabe-Ito colour-blind-safe palette (same as main8_main STEP 6a)
col.black  = [0 0 0];       col.orange = [230 159   0]/255;
col.green  = [  0 158 115]/255; col.yellow = [240 228  66]/255;
col.blue   = [  0 114 178]/255; col.vermil = [213  94   0]/255;
col.purple = [204 121 167]/255; col.gray   = [0.5 0.5 0.5];
N_sym_col  = {col.blue, col.orange, col.green, col.vermil, col.purple, col.yellow};

% Average over UE realisations (dim 3) and Monte-Carlo trials (dim 2) -> [SNR x N_sym_cases]
Cseq_num_avg  = squeeze(mean(mean(cap_seq_numerical_all,    2, 'omitnan'), 3, 'omitnan'));
Cseq_theo_avg = squeeze(mean(mean(cap_seq_theorectical_all, 2, 'omitnan'), 3, 'omitnan'));
Cnet_num_avg  = squeeze(mean(mean(cap_net_numerical_all,    2, 'omitnan'), 3, 'omitnan'));
Cnet_theo_avg = squeeze(mean(mean(cap_net_theorectical_all, 2, 'omitnan'), 3, 'omitnan'));

% ---- Figure 1: Cbar_seq vs. SNR, (eq_cap_seq_avg) --------------------
figure('Color','w','Units','inches','Position',[1 1 7 5]);
hold on; box on; grid on;
for n = 1:N_sym_cases
    c = N_sym_col{mod(n-1, numel(N_sym_col)) + 1};
    plot(SNR_Tx_db, Cseq_num_avg(:,n), 'o', 'Color', c, 'MarkerFaceColor', c, ...
        'MarkerSize', 6, 'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', N_sym_list(n)));
    plot(SNR_Tx_db, Cseq_theo_avg(:,n), '--', 'Color', c, 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
plot(nan, nan, 'o', 'Color', col.black, 'MarkerFaceColor', col.black, ...
    'DisplayName', 'Num.: Monte-Carlo (realized channel)');
plot(nan, nan, '--', 'Color', col.black, 'LineWidth', 1.5, ...
    'DisplayName', 'Theo.: closed-form (Gamma interference)');
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
ylabel('$\bar{\mathcal{C}}_{\rm seq}$ (bits/sequence)');
title('Ensemble-Average Capacity per Sequence vs. SNR');
legend('Location','northwest','Box','on','FontSize',10,'NumColumns',2);
savefig(gcf, fullfile(fig_dir,'fig_cap_seq_snr_nsym.fig'));
exportgraphics(gcf, fullfile(fig_dir,'fig_cap_seq_snr_nsym.png'), 'Resolution',300);

% ---- Figure 2: Cbar_net vs. SNR, (eq_cap_net) -------------------------
figure('Color','w','Units','inches','Position',[1 1 7 5]);
hold on; box on; grid on;
for n = 1:N_sym_cases
    c = N_sym_col{mod(n-1, numel(N_sym_col)) + 1};
    plot(SNR_Tx_db, Cnet_num_avg(:,n), 'o', 'Color', c, 'MarkerFaceColor', c, ...
        'MarkerSize', 6, 'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', N_sym_list(n)));
    plot(SNR_Tx_db, Cnet_theo_avg(:,n), '--', 'Color', c, 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
plot(nan, nan, 'o', 'Color', col.black, 'MarkerFaceColor', col.black, ...
    'DisplayName', 'Num.: Monte-Carlo (realized channel)');
plot(nan, nan, '--', 'Color', col.black, 'LineWidth', 1.5, ...
    'DisplayName', 'Theo.: closed-form (Gamma interference)');
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
ylabel('$\bar{\mathcal{C}}_{\rm net}$ (bits/s)');
title('Ensemble-Average Net Capacity vs. SNR');
legend('Location','northwest','Box','on','FontSize',10,'NumColumns',2);
savefig(gcf, fullfile(fig_dir,'fig_cap_net_snr_nsym.fig'));
exportgraphics(gcf, fullfile(fig_dir,'fig_cap_net_snr_nsym.png'), 'Resolution',300);

fprintf('\nCapacity figures saved: fig_cap_seq_snr_nsym.png, fig_cap_net_snr_nsym.png (in %s)\n', fig_dir);


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function pos = rand_pos_in_roi(roi, N)
    pos = zeros(3, N);
    for dim = 1:3
        lo = roi(dim,1);  hi = roi(dim,2);
        if lo == hi
            pos(dim,:) = lo;
        else
            pos(dim,:) = lo + (hi - lo) .* rand(1, N);
        end
    end
end

function BER = func_ana_ber(H_com_los_all, H_com_nlos_all, C_w_time, Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block)
    % Semi-analytical BER: instantaneous SINR from the geometry-determined
    % channel realisation, mapped through the Gaussian Q-function and averaged
    % over blocks/symbols. Averaging over the outer Monte-Carlo scene loop then
    % yields the expected BER (deterministic-per-path model, no parametric
    % interference distribution). Unused in this file's sweep (kept for parity
    % with main7_comm_analysis.m); func_ana_ber_closeform is used instead.
    BER = zeros(N_block, N_sym_per_block);
    for i = 1:N_block
        h_com_los_i  = H_com_los_all(i, :, :);
        h_com_nlos_i = H_com_nlos_all(i, :, :, :);
        for j = 1:N_sym_per_block
            idx_d = ((j-1)*2 + 1)*L_slot_per_sym + 1 : ((j-1)*2 + 2)*L_slot_per_sym;
            p = C_w_time(idx_d);          % p_prbs_even_j (column vector)
            ph_p = p' * p;                % scalar: p^H * p = ||p||^2  (real, positive)

            % Numerator:  |h_los|^2 * |p^H p|^2
            signal_component = abs(h_com_los_i)^2 * abs(ph_p)^2;

            % Denominator (interference sum):  sum_k |h_nlos^(k)|^2 * |p^H p|^2
            interference_component = 0;
            for k = 1:N_ref_tars
                h_nlos_k = h_com_nlos_i(:, :, k);
                interference_component = interference_component + abs(h_nlos_k)^2 * abs(ph_p)^2;
            end

            % Denominator (noise):  sigma_n^2 * ||p||^2
            noise_component = Noise_power_com * real(ph_p);

            SINR_i_j = signal_component / (interference_component + noise_component);
            BER(i, j) = qfunc(sqrt(2 * SINR_i_j));
        end
    end
    BER = mean(BER(:));
end

function [Net_throughput] = func_ana_net(H_com_los_all, H_com_nlos_all, C_w_time, Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block, R_b, r_c)

    BER = zeros(N_block, N_sym_per_block);
    for i = 1:N_block
        h_com_los_i  = H_com_los_all(i, :, :);
        h_com_nlos_i = H_com_nlos_all(i, :, :, :);
        for j = 1:N_sym_per_block
            idx_d = ((j-1)*2 + 1)*L_slot_per_sym + 1 : ((j-1)*2 + 2)*L_slot_per_sym;
            p = C_w_time(idx_d);          % p_prbs_even_j (column vector)
            ph_p = p' * p;                % scalar: p^H * p = ||p||^2  (real, positive)

            % Numerator:  |h_los|^2 * |p^H p|^2
            signal_component = abs(h_com_los_i)^2 * abs(ph_p)^2;

            % Denominator (interference sum):  sum_k |h_nlos^(k)|^2 * |p^H p|^2
            interference_component = 0;
            for k = 1:N_ref_tars
                h_nlos_k = h_com_nlos_i(:, :, k);
                interference_component = interference_component + abs(h_nlos_k)^2 * abs(ph_p)^2;
            end

            % Denominator (noise):  sigma_n^2 * ||p||^2
            noise_component = Noise_power_com * real(ph_p);

            SINR_i_j = signal_component / (interference_component + noise_component);
            BER(i, j) = qfunc(sqrt(2 * SINR_i_j));
        end
    end
    R_sym = r_c * R_b * (1 - BER);
    Net_throughput = mean(R_sym(:));  % [bits/s]
end

function [Cbar_seq, Cbar_net] = func_cap_seq(H_com_los_all, H_com_nlos_all, C_w_time, ...
        Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block, T_prbs)
    % NUMERICAL ensemble-average capacity, (eq_cap_seq)-(eq_cap_net)
    % (5_7_results_comm.tex): evaluated directly from the TRUE geometry-
    % realized channel of this Monte-Carlo trial (same per-realization
    % philosophy, and same gamma_{k,i} components, as func_ana_ber /
    % (eq_sinr_k_i_b) -- i.e. WITH the L^2/L despreading-gain structure,
    % unlike the old func_ana_cap's per-chip, gain-free SINR).
    %
    % i (1..N_block) indexes the N_prbs MLS sequences of the frame;
    % j (1..N_sym_per_block) indexes the N_bit_prbs data slots within a
    % sequence, matching (eq_cap_seq)'s k-index.
    C_seq_per_block = zeros(N_block, 1);   % C_seq^{(i)}
    for i = 1:N_block
        h_com_los_i  = H_com_los_all(i, :, :);
        h_com_nlos_i = H_com_nlos_all(i, :, :, :);
        C_sum = 0;
        for j = 1:N_sym_per_block
            idx_d = ((j-1)*2 + 1)*L_slot_per_sym + 1 : ((j-1)*2 + 2)*L_slot_per_sym;
            p    = C_w_time(idx_d);       % p_prbs_even_j (column vector)
            ph_p = p' * p;                 % ||p||^2 = L (scalar, real)

            % Signal: |h_los|^2 * L^2
            signal_component = abs(h_com_los_i)^2 * abs(ph_p)^2;

            % Interference: sum_k |h_nlos_k|^2 * L^2 (perfect-overlap convention,
            % consistent with func_ana_ber/func_ana_ber_closeform)
            interference_component = 0;
            for k = 1:N_ref_tars
                h_nlos_k = h_com_nlos_i(:, :, k);
                interference_component = interference_component + abs(h_nlos_k)^2 * abs(ph_p)^2;
            end

            % Noise: sigma_com^2 * L  (matches (eq_sinr_k_i_b), NOT L^2)
            noise_component = Noise_power_com * real(ph_p);

            gamma_kj = signal_component / (interference_component + noise_component);
            C_sum = C_sum + log2(1 + gamma_kj);   % one term of (eq_cap_seq)
        end
        C_seq_per_block(i) = C_sum;                % C_seq^{(i)}
    end
    Cbar_seq = mean(C_seq_per_block);  % (eq_cap_seq_avg), [bits/sequence]
    Cbar_net = Cbar_seq / T_prbs;       % (eq_cap_net), [bits/s]
end

function [Cbar_seq, Cbar_net] = func_ana_cap_closeform(P_tx, G_tx, G_ue, L_slot_per_sym, N_bit_prbs, N_ref_tars, ...
        Noise_power_com, UE_position, Lambda, N_tars, N_scats, Region_of_interest, T_prbs)
    % THEORETICAL (closed-form) ensemble-average capacity: E[log2(1+gamma)]
    % taken directly over the Gamma-distributed aggregate NLOS interference
    % P_nlos ~ Gamma(K, Omega), then scaled by N_bit_prbs data slots per
    % sequence and normalized by T_prbs, per (eq_cap_seq_avg) and
    % (eq_cap_net). Uses the SAME P_los/Omega geometry-based derivation
    % as func_ana_ber_closeform (Steps 1-2 identical), but the reduction here
    % is a direct expectation over P_nlos -- no Jacobian/change-of-variables
    % is needed since log2(1+gamma) is evaluated as a function of P_nlos
    % directly, unlike the BER's Q(sqrt(2*gamma)) which had to be inverted.

    % ---- 1. System Parameters & Constants (identical to func_ana_ber_closeform) ----
    L = L_slot_per_sym;
    L_sq = L^2;
    A_e = G_ue * (Lambda^2) / (4 * pi);

    r_ue = norm(UE_position);
    PL_los = 1 / (4 * pi * r_ue^2);
    C = PL_los * P_tx * G_tx * A_e * L_sq;   % P_los * L^2
    N = Noise_power_com * L;                  % sigma_com^2 * L

    % ---- 2. Pure Spatial Statistics via Deterministic Grid Integration ----
    % (estimate Omega = E[P_nlos^(m)] from the scene geometry; identical to
    % func_ana_ber_closeform.)
    Nx = 100; Ny = 200;
    x_vec = linspace(Region_of_interest(1,1), Region_of_interest(1,2), Nx);
    y_vec = linspace(Region_of_interest(2,1), Region_of_interest(2,2), Ny);
    [X, Y] = meshgrid(x_vec, y_vec);

    A_ROI = (Region_of_interest(1,2) - Region_of_interest(1,1)) * ...
            (Region_of_interest(2,2) - Region_of_interest(2,1));
    dA = A_ROI / (Nx * Ny);
    rho = (N_tars + N_scats) / A_ROI;

    R_k_ue = sqrt((X - UE_position(1)).^2 + (Y - UE_position(2)).^2);
    R_tx_k = sqrt(X.^2 + Y.^2);

    [R_k_ue_sorted, sort_idx] = sort(R_k_ue(:), 'ascend');
    R_tx_k_sorted = R_tx_k(sort_idx);

    target_weight_per_cell = rho * dA;
    cum_targets = cumsum(ones(size(R_k_ue_sorted)) * target_weight_per_cell);

    K = N_ref_tars;
    idx_needed = find(cum_targets >= K, 1, 'first');
    if isempty(idx_needed), idx_needed = length(cum_targets); end

    target_ticks = (1:K) - 0.5;
    r_k_ue_virtual = interp1(cum_targets(1:idx_needed), R_k_ue_sorted(1:idx_needed), target_ticks, 'linear', 'extrap');
    r_tx_k_virtual = interp1(cum_targets(1:idx_needed), R_tx_k_sorted(1:idx_needed), target_ticks, 'linear', 'extrap');

    E_sigma_tar  = 10 / log(2);
    E_sigma_scat = 5 / log(2);
    E_sigma_rcs  = (N_tars / (N_tars + N_scats)) * E_sigma_tar + ...
                   (N_scats / (N_tars + N_scats)) * E_sigma_scat;

    PL_nlos_virtual = 1 ./ ((4 * pi * r_tx_k_virtual.^2) .* (4 * pi * r_k_ue_virtual.^2));
    nlos_powers_virtual = E_sigma_rcs * PL_nlos_virtual * P_tx * G_tx * A_e * L_sq;

    Omega = mean(nlos_powers_virtual);

    % ---- 3. E[log2(1+gamma)] via direct expectation over P_nlos ~ Gamma(K,Omega) ----
    x_max  = gaminv(1 - 1e-12, K, Omega);   % effectively-full Gamma support
    x_vals = linspace(0, x_max, 20000);

    f_I     = (x_vals.^(K-1) .* exp(-x_vals./Omega)) ./ (Omega^K * gamma(K));
    gamma_x = C ./ (x_vals + N);
    integrand = log2(1 + gamma_x) .* f_I;

    E_log2_1p_gamma = trapz(x_vals, integrand);

    % ---- 4. Scale by N_bit_prbs data slots, normalize by sequence duration ----
    Cbar_seq = N_bit_prbs * E_log2_1p_gamma;  % (eq_cap_seq_avg), [bits/sequence]
    Cbar_net = Cbar_seq / T_prbs;               % (eq_cap_net), [bits/s]
end

function avg_ber = func_ana_ber_closeform(P_tx, G_tx, G_ue, L_slot_per_sym, N_ref_tars, ...
                                                Noise_power_com, UE_position, Lambda, ...
                                                N_tars, N_scats, Region_of_interest)
    % UNCHANGED vs. main7_comm_analysis.m -- kept verbatim for the BER sweep.
    % =====================================================================
    % 1. System Parameters & Constants
    % =====================================================================
    L = L_slot_per_sym;
    L_sq = L^2;
    A_e = G_ue * (Lambda^2) / (4 * pi);

    % LOS Power (C)
    r_ue = norm(UE_position);
    PL_los = 1 / (4 * pi * r_ue^2);
    C = PL_los * P_tx * G_tx * A_e * L_sq;

    % Noise Power (N)
    N = Noise_power_com * L;

    % =====================================================================
    % 2. Pure Spatial Statistics via Deterministic Grid Integration
    % =====================================================================
    % Define a fine grid over the actual 2D ROI to capture boundaries perfectly
    Nx = 100; Ny = 200;
    x_vec = linspace(Region_of_interest(1,1), Region_of_interest(1,2), Nx);
    y_vec = linspace(Region_of_interest(2,1), Region_of_interest(2,2), Ny);
    [X, Y] = meshgrid(x_vec, y_vec);

    A_ROI = (Region_of_interest(1,2) - Region_of_interest(1,1)) * ...
            (Region_of_interest(2,2) - Region_of_interest(2,1));
    dA = A_ROI / (Nx * Ny);
    rho = (N_tars + N_scats) / A_ROI;

    % Distance from every point in the ROI to the UE and the TX (origin)
    R_k_ue = sqrt((X - UE_position(1)).^2 + (Y - UE_position(2)).^2);
    R_tx_k = sqrt(X.^2 + Y.^2);

    % Sort all grid cells by distance to the UE
    [R_k_ue_sorted, sort_idx] = sort(R_k_ue(:), 'ascend');
    R_tx_k_sorted = R_tx_k(sort_idx);

    % Each grid cell contains an expected number of targets equal to rho * dA
    target_weight_per_cell = rho * dA;
    cum_targets = cumsum(ones(size(R_k_ue_sorted)) * target_weight_per_cell);

    % Find the cells needed to capture exactly N_ref_tars (K) closest targets
    K = N_ref_tars;
    idx_needed = find(cum_targets >= K, 1, 'first');
    if isempty(idx_needed), idx_needed = length(cum_targets); end

    % Map the continuous spatial cells into K discrete target power steps
    % to exactly replicate the discrete sorting of your closeform function
    target_ticks = (1:K) - 0.5;
    r_k_ue_virtual = interp1(cum_targets(1:idx_needed), R_k_ue_sorted(1:idx_needed), target_ticks, 'linear', 'extrap');
    r_tx_k_virtual = interp1(cum_targets(1:idx_needed), R_tx_k_sorted(1:idx_needed), target_ticks, 'linear', 'extrap');

    % Analytical Ensembled Radar Cross Section E[sigma_rcs]
    E_sigma_tar  = 10 / log(2);
    E_sigma_scat = 5 / log(2);
    E_sigma_rcs  = (N_tars / (N_tars + N_scats)) * E_sigma_tar + ...
                   (N_scats / (N_tars + N_scats)) * E_sigma_scat;

    % Analytical Path Loss and Power vectors for the K closest items
    PL_nlos_virtual = 1 ./ ((4 * pi * r_tx_k_virtual.^2) .* (4 * pi * r_k_ue_virtual.^2));
    nlos_powers_virtual = E_sigma_rcs * PL_nlos_virtual * P_tx * G_tx * A_e * L_sq;

    % Core statistical parameters matching closeform logic
    Omega = mean(nlos_powers_virtual);

    % =====================================================================
    % 3. Calculate True Statistical Boundaries
    % =====================================================================
    % Max interference occurs when all K paths align perfectly in phase
    I_max = (sum(sqrt(nlos_powers_virtual)))^2;

    p_min = qfunc(sqrt(2 * C / N));               % Best-case (Zero interference)
    p_max = qfunc(sqrt(2 * C / (I_max + N)));     % Worst-case (Max constructive interference)


    % Saturation Guard
    if p_min >= 0.5
        avg_ber = 0.5;
        return;
    end

    % =====================================================================
    % 4. Numerical Integration (Jacobian Transformation Domain)
    % =====================================================================
    p_values = linspace(p_min, p_max, 10000);

    % Gamma Distribution PDF for Interference
    f_I = @(x) (x.^(K-1) .* exp(-x./Omega)) ./ (Omega^K * gamma(K));

    % Mapping BER back to Interference power levels (Inverse Q-function)
    q_inv = qfuncinv(p_values);
    I_p = (2 * C) ./ (q_inv.^2) - N;

    % Jacobian transformation derivative |dI/dp|
    jacobian = (4 * sqrt(2*pi) * C) ./ (q_inv.^3) .* exp((q_inv.^2)/2);

    % Compute mapped PDF across the bounded error vector
    pdf_values = f_I(I_p) .* jacobian;

    % Compute Final Expectation via Trapezoidal Integration
    avg_ber = trapz(p_values, p_values .* pdf_values);
end

