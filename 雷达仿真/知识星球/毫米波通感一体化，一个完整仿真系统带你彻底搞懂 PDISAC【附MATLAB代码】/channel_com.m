clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

% =====================================================================
%  SHARED CONFIGURATION AND CHANNEL STATISTICS SETTINGS
% =====================================================================
main1_matlab_config;
G_tx_db = channel_statistics_cfg.system_override.gain_tx_db;
G_rx_db = channel_statistics_cfg.system_override.gain_rx_db;
G_ue_db = channel_statistics_cfg.system_override.gain_ue_db;
G_tx = 10^(G_tx_db/10); G_rx = 10^(G_rx_db/10); G_ue = 10^(G_ue_db/10);
N_ran = channel_statistics_cfg.random_scenes;
Mc = channel_statistics_cfg.monte_carlo;
N_sym_list = channel_statistics_cfg.symbols_per_block(:).';
N_sym_cases = length(N_sym_list);
SNR_Tx_db = channel_statistics_cfg.snr_db(:).';
SNR_Tx = 10.^(SNR_Tx_db / 10);

N_samples_per_trial = N_block + 1;
N_trials_total = N_ran * N_sym_cases * length(SNR_Tx) * Mc;
N_samples_total = N_trials_total * N_samples_per_trial;

los_channel_all     = complex(zeros(N_samples_total, 1));
nlos_channel_all    = complex(zeros(N_samples_total, 1));
overall_channel_all = complex(zeros(N_samples_total, 1));

trial_summary = array2table(nan(N_trials_total, 8), 'VariableNames', { ...
    'Trial_ID', ...
    'RAN_Index', ...
    'N_sym_per_block', ...
    'SNR_dB', ...
    'MC_Index', ...
    'Mean_LOS_Power_dB', ...
    'Mean_NLOS_Power_dB', ...
    'Mean_Overall_Power_dB'});

trial_id = 0;
sample_ptr = 1;

% =====================================================================
%  LOOP 1 – random UE realisation
% =====================================================================
for idx_ran = 1:N_ran

    fprintf('\n========== UE Realisation %d / %d ==========\n', idx_ran, N_ran);

    N_ue        = 1;
    UE_position = rand_pos_in_roi(Region_of_interest, N_ue);
    UE_vel      = VEL_RANGE_UE * (2*randn(3, N_ue) - 1);
    UE_vel(3)   = 0;
    UE_rcs      = 10.^( (10*log10(10) + (10*log10(20)-10*log10(10))*randn(1)) / 10 );

    Rel_positions  = UE_position - Tx_position;
    [~, ~, Range_Tx_UE] = cart2sph(Rel_positions(1), Rel_positions(2), Rel_positions(3));
    Radial_UE_vel  = dot(UE_vel, Rel_positions / norm(Rel_positions));

    % ==================================================================
    %  LOOP 2 – N_sym_per_block sweep
    % ==================================================================
    for idx_nsym = 1:N_sym_cases

        N_sym_per_block = N_sym_list(idx_nsym);

        L_slot_per_sym = N_chip / (2 * N_sym_per_block);
        N_slot         = N_chip / L_slot_per_sym;

        fprintf('\n  ----- N_sym_per_block = %d -----\n', N_sym_per_block);

        % ================================================================
        %  LOOP 3 – SNR sweep
        % ================================================================
        for idx_snr = 1:length(SNR_Tx)
            snr_tx = SNR_Tx(idx_snr);
            P_tx   = Noise_power_com * snr_tx;

            % ============================================================
            %  LOOP 4 – Monte-Carlo trials
            % ============================================================
            for idx_mc = 1:Mc

                N_tars        = 5;
                Tars_position = rand_pos_in_roi(Region_of_interest, N_tars);
                Tars_vel      = VEL_RANGE_TAR * (2*randn(3, N_tars) - 1);
                Tars_vel(3,:) = 0;
                Tars_rcs      = 10.^( (10*log10(10) + ...
                                       (10*log10(20)-10*log10(10))*randn(1, N_tars)) / 10 );

                N_tars        = N_tars + 1;
                Tars_position = [Tars_position, UE_position];
                Tars_vel      = [Tars_vel,      UE_vel];
                Tars_rcs      = [Tars_rcs,      UE_rcs];

                Tars_motion = phased.Platform( ...
                    'InitialPosition', Tars_position, ...
                    'Velocity',        Tars_vel);
                UE_motion = phased.Platform( ...
                    'InitialPosition', UE_position, ...
                    'Velocity',        UE_vel);

                N_scats        = 20;
                Scats_position = func_generate_static_scatterers(N_scats, Region_of_interest);
                Scats_vel      = zeros(size(Scats_position));
                Scats_rcs      = 10.^( (10*log10(5) + ...
                                        (10*log10(10)-10*log10(5))*randn(1, N_scats)) / 10 );
                Scats_motion   = phased.Platform( ...
                    'InitialPosition', Scats_position, ...
                    'Velocity',        Scats_vel);

                N_ref_tars = 10;

                main3_signal_channel_model;
                
                H_los = squeeze(H_com_los_all);
                H_nlos_total = squeeze(sum(H_com_nlos_all, 4));

                H_los = H_los(:);
                H_nlos_total = H_nlos_total(:);
                H_overall = H_los + H_nlos_total;

                idx_range = sample_ptr:(sample_ptr + N_samples_per_trial - 1);
                los_channel_all(idx_range)     = H_los;
                nlos_channel_all(idx_range)    = H_nlos_total;
                overall_channel_all(idx_range) = H_overall;
                sample_ptr = sample_ptr + N_samples_per_trial;

                trial_id = trial_id + 1;
                trial_summary{trial_id, :} = [ ...
                    trial_id, ...
                    idx_ran, ...
                    N_sym_per_block, ...
                    SNR_Tx_db(idx_snr), ...
                    idx_mc, ...
                    mean(channel_power_db(H_los), 'omitnan'), ...
                    mean(channel_power_db(H_nlos_total), 'omitnan'), ...
                    mean(channel_power_db(H_overall), 'omitnan')];
            end

            fprintf('    Nsym=%3d | r_ue=%5.1f m | v_ue=%+5.1f m/s | SNR=%+3d dB | Trials=%d/%d\n', ...
                N_sym_per_block, Range_Tx_UE, Radial_UE_vel, SNR_Tx_db(idx_snr), ...
                trial_id, N_trials_total);

        end
    end
end

los_channel_all     = los_channel_all(1:(sample_ptr-1));
nlos_channel_all    = nlos_channel_all(1:(sample_ptr-1));
overall_channel_all = overall_channel_all(1:(sample_ptr-1));
trial_summary       = trial_summary(1:trial_id, :);

los_power_db     = channel_power_db(los_channel_all);
nlos_power_db    = channel_power_db(nlos_channel_all);
overall_power_db = channel_power_db(overall_channel_all);

results_dir = fullfile(script_dir, 'exported_statistics');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

mat_filename = fullfile(results_dir, 'statistics_channel.mat');
csv_filename = fullfile(results_dir, 'statistics_channel_summary.csv');

save(mat_filename, ...
    'los_channel_all', ...
    'nlos_channel_all', ...
    'overall_channel_all', ...
    'los_power_db', ...
    'nlos_power_db', ...
    'overall_power_db', ...
    'trial_summary', ...
    'N_block', ...
    'N_ref_tars', ...
    'N_ran', ...
    'Mc', ...
    'N_sym_list', ...
    'SNR_Tx_db', ...
    '-v7.3');
writetable(trial_summary, csv_filename);

fprintf('\nSaved channel samples to %s\n', mat_filename);
fprintf('Saved trial summary to %s\n', csv_filename);

% =====================================================================
%  FIGURE 1 – Power (dB) distributions
% =====================================================================
fig = figure('Color', 'w', 'Name', 'Communication Channel Distributions');
tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

plot_channel_distribution(los_power_db, ...
    'LOS Communication Channel', [0.00 0.45 0.74]);
plot_channel_distribution(nlos_power_db, ...
    'NLOS Communication Channel', [0.85 0.33 0.10]);
plot_channel_distribution(overall_power_db, ...
    'Overall Communication Channel', [0.47 0.67 0.19]);

sgtitle('Communication Channel Power Distributions Across Trials');

fig_dir = fullfile(script_dir, 'fig_exported');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

savefig(fig, fullfile(fig_dir, 'fig_com_channel_distribution_v2.fig'));
fig_filename = fullfile(fig_dir, 'fig_com_channel_distribution_v2.png');
exportgraphics(fig, fig_filename, 'Resolution', 300);
fprintf('Saved figure to %s\n', fig_filename);

% =====================================================================
%  FIGURE 2 – Raw (real / imaginary) distributions
% =====================================================================
fig2 = figure('Color', 'w', 'Name', 'Communication Channel Raw Distributions');
tiledlayout(fig2, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

plot_channel_raw_distribution(channel_raw_part(los_channel_all, 'real'), ...
    'LOS - Real Part', [0.00 0.45 0.74]);
plot_channel_raw_distribution(channel_raw_part(nlos_channel_all, 'real'), ...
    'NLOS - Real Part', [0.85 0.33 0.10]);
plot_channel_raw_distribution(channel_raw_part(overall_channel_all, 'real'), ...
    'Overall - Real Part', [0.47 0.67 0.19]);

plot_channel_raw_distribution(channel_raw_part(los_channel_all, 'imag'), ...
    'LOS - Imag Part', [0.00 0.45 0.74]);
plot_channel_raw_distribution(channel_raw_part(nlos_channel_all, 'imag'), ...
    'NLOS - Imag Part', [0.85 0.33 0.10]);
plot_channel_raw_distribution(channel_raw_part(overall_channel_all, 'imag'), ...
    'Overall - Imag Part', [0.47 0.67 0.19]);

sgtitle('Communication Channel Raw (Real/Imaginary) Distributions');

savefig(fig2, fullfile(fig_dir, 'fig_com_channel_raw_distribution_v2.fig'));
fig2_filename = fullfile(fig_dir, 'fig_com_channel_raw_distribution_v2.png');
exportgraphics(fig2, fig2_filename, 'Resolution', 300);
fprintf('Saved figure to %s\n', fig2_filename);

% =====================================================================
%  LOCAL FUNCTIONS
% =====================================================================
function pos = rand_pos_in_roi(roi, N)
    pos = zeros(3, N);
    for dim = 1:3
        lo = roi(dim,1);
        hi = roi(dim,2);
        if lo == hi
            pos(dim,:) = lo;
        else
            pos(dim,:) = lo + (hi - lo) .* randn(1, N);
        end
    end
end

function power_db = channel_power_db(h)
    power_db = 10 * log10(abs(h).^2 + eps);
    power_db = power_db(isfinite(power_db));
end

function plot_channel_distribution(power_db, plot_title, color)
    nexttile;
    histogram(power_db, 'Normalization', 'pdf', ...
        'FaceColor', color, ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.75);
    grid on;
    box on;
    xlabel('Channel power (dB)');
    ylabel('PDF');
    title(plot_title);
end

function part_val = channel_raw_part(h, part_type)
    if strcmpi(part_type, 'real')
        part_val = real(h);
    else
        part_val = imag(h);
    end
    part_val = part_val(isfinite(part_val));
end

function plot_channel_raw_distribution(raw_val, plot_title, color)
    nexttile;
    % 'pdf' normalization only rescales the histogram's y-axis (area = 1)
    % for readability -- the underlying x-data (raw_val) is untouched.
    histogram(raw_val, 'Normalization', 'pdf', ...
        'FaceColor', color, ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.75);
    grid on;
    box on;
    xlabel('Amplitude (linear)');
    ylabel('PDF');
    title(plot_title);
end
