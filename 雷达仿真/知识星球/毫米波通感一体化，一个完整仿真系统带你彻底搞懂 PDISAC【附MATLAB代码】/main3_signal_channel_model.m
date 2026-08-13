% SIGNAL_CHANNEL_MODEL  (consolidated Group 3: signal and channel modelling)
% -------------------------------------------------------------------------
% Builds the transmit ISAC waveform and propagates it through the sensing and
% communication channels to form the received matrices.
% Shared kernel helpers kept standalone: h_sen_tars, func_get_x_delay,
%     func_get_array_response.
% Prerequisite in the workspace: main2_topology (scene) + main1_matlab_config constants.
% Set NO_SIGNAL_PLOT=true to skip the transmit-signal figure.

%% Transmit ------------------------------------------------------------------


C_w_time = func_generate_waveform(N_chip); % shape (N_chip x 1) 
Data_Tx_bin = randi([0 1], [N_block N_sym_per_block N_tx_ant]); % binary data
Data_Tx = zeros(N_block, N_chip, N_tx_ant);
Data_Tx_no_data = zeros(N_block, N_chip, N_tx_ant);


for i = 1:N_block
    Prbs_block_i = C_w_time;
    Data_Tx_bin_block_i = Data_Tx_bin(i, :, :);
    BPSK_block_i = func_bpsk_mod(Data_Tx_bin_block_i);

    Data_Tx_no_data(i, :, :) = C_w_time;

    sym_idx = 1;

    for slot = 1:N_slot

        idx = (slot-1)*L_slot_per_sym + 1 : slot*L_slot_per_sym;

        if mod(slot,2)==0  % even slot
            Prbs_block_i(idx) = Prbs_block_i(idx) * BPSK_block_i(sym_idx);
            sym_idx = sym_idx + 1;
        end
    end

    Data_Tx(i, :, :) = Prbs_block_i;
end
%% Receive -------------------------------------------------------------------

Data_Tx_pad = cat(1, Data_Tx, zeros(1, N_chip, N_tx_ant));  % (N_block+1, N_chip, N_tx_ant)

Data_Rx = zeros(N_block, N_chip, N_rx_ant); % ISAC received signal at Rx
Data_Rx_no_noise = zeros(N_block, N_chip, N_rx_ant); % ISAC received signal at Rx no noise
Data_Rx_no_data = zeros(N_block, N_chip, N_rx_ant); % ISAC received signal at Rx no data
Data_UE = zeros(N_block, N_chip, N_ue_ant); % ISAC received signal at UE

H_com_los_all = zeros(N_block+1, N_ue_ant, N_tx_ant);
H_com_nlos_all = zeros(N_block+1, N_ue_ant, N_tx_ant, N_ref_tars);

reset(Tars_motion);
reset(UE_motion);

for i = 1:N_block

    T_step = i * T_pmcw;
    Current_Tars_position = Tars_position;
    Current_Tars_vel      = Tars_vel;
    Current_Scats_position = Scats_position;
    Current_Scats_vel      = Scats_vel;
    Current_UE_position = UE_position;
    Current_UE_vel      = UE_vel;
    
    %---------%
    % Sensing %
    %---------%
    % Moving Target Sensing channel (N_rx_ant x N_tx_ant x N_tars)
    [H_sen_tars, Range_tars, Radial_tars_vel] = h_sen_tars(N_tars, Current_Tars_position, Tars_rcs, Current_Tars_vel, Tx_position, N_tx_ant, Rx_position, N_rx_ant, P_tx, G_tx, G_rx, Lambda, fc, T_step);
    % Static Targets (Scatters) Sensing channel (N_rx_ant x N_tx_ant x N_scaters)
    [H_sen_scats, Range_scats, Radial_scats_vel] = h_sen_tars(N_scats, Current_Scats_position, Scats_rcs, Current_Scats_vel, Tx_position, N_tx_ant, Rx_position, N_rx_ant, P_tx, G_tx, G_rx, Lambda, fc, T_step);

    % X with delay
    Delay_tars  = 2 * Range_tars  / c;
    Delay_scats = 2 * Range_scats / c;
    X_t_tau_tars  = func_get_x_delay(Data_Tx, i, Delay_tars,  T_chip); % N_chip x N_tx_ant x N_tars
    X_t_tau_scats = func_get_x_delay(Data_Tx, i, Delay_scats, T_chip); % N_chip x N_tx_ant x N_scats

    H_sen = cat(3, H_sen_tars, H_sen_scats);       % N_rx_ant x N_tx_ant x (N_tars + N_scats)
    X_t_tau_sen = cat(3, X_t_tau_tars, X_t_tau_scats);   % N_chip x N_tx_ant x (N_tars + N_scats)

    H_mat_sen = reshape(H_sen, N_rx_ant, N_tx_ant * (N_tars + N_scats)); % N_rx_ant x N_tx_ant * (N_tars + N_scats)
    X_mat_sen = reshape(permute(X_t_tau_sen, [2, 3, 1]), (N_tars + N_scats) * N_tx_ant, N_chip); %  N_tx_ant * (N_tars + N_scats) x N_chip

    Y_t_sen = H_mat_sen * X_mat_sen; % (N_rx_ant, N_chip)
    Y_t_sen = func_add_noise(Y_t_sen, Noise_power_sen);
    
    % Data Rx with noise
    Data_Rx(i, :,:) = Y_t_sen.';

    %------------------------------%
    % Sensing no noise and no data %
    %------------------------------%
    % X with delay
    Delay_tars  = 2 * Range_tars  / c;
    Delay_scats = 2 * Range_scats / c;
    X_t_tau_tars  = func_get_x_delay(Data_Tx_no_data, i, Delay_tars,  T_chip); % N_chip x N_tx_ant x N_tars
    X_t_tau_scats = func_get_x_delay(Data_Tx_no_data, i, Delay_scats, T_chip); % N_chip x N_tx_ant x N_scats
    X_t_tau_sen = cat(3, X_t_tau_tars, X_t_tau_scats);   % N_chip x N_tx_ant x (N_tars + N_scats)
    X_mat_sen = reshape(permute(X_t_tau_sen, [2, 3, 1]), (N_tars + N_scats) * N_tx_ant, N_chip); %  N_tx_ant * (N_tars + N_scats) x N_chip
    Y_t_sen = H_mat_sen * X_mat_sen; % (N_rx_ant, N_chip)
    
    % Data Rx without noise, without data
    Data_Rx_no_noise(i, :,:) = Y_t_sen.';
    % Data Rx with noise, without data
    Data_Rx_no_data(i, :, :) = func_add_noise(Y_t_sen, Noise_power_sen).';
    

    %---------------%
    % Communication %
    %---------------%
    [H_com_los, Range_UE, Radial_Current_UE_vel] = h_com_los(N_ue, Current_UE_position, Current_UE_vel, N_ue_ant, Tx_position, N_tx_ant, P_tx, G_tx, G_ue, Lambda, fc, T_step);
    H_com_los_all(i, :,:) = H_com_los;

    
    All_tars_position = [Current_Tars_position(:, 1:end-1)  Current_Scats_position];
    All_radial_tars_vel = [Radial_tars_vel(:, 1:end-1) Radial_scats_vel];
    All_radial_tars_rcs = [Tars_rcs(:, 1:end-1)  Scats_rcs];

    % Com NLoS channel [Tx → reflected targets → UE] (N_ue_ant x N_tx_ant x N_ref_tars)
    [H_com_nlos, Range_Tx_ref_tars, Range_ref_tars_UE, Radial_ref_tars_vel] = h_com_nlos(N_ref_tars, All_tars_position, All_radial_tars_vel, All_radial_tars_rcs, Tx_position, N_tx_ant, Current_UE_position, N_ue_ant, Radial_Current_UE_vel, P_tx, G_tx, G_ue, Lambda, fc, T_step);
    H_com_nlos_all(i, :, :, :) = H_com_nlos;
    
    Delay_los = Range_UE / c;
    Delay_nlos = (Range_Tx_ref_tars + Range_ref_tars_UE) / c;
    X_t_tau_los = func_get_x_delay(Data_Tx_pad, i, Delay_los,  T_chip); % N_chip x N_tx_ant
    X_t_tau_nlos = func_get_x_delay(Data_Tx_pad, i, Delay_nlos, T_chip); % N_chip x N_tx_ant x N_ref_tars

    % LOS contribution (N_ue_ant, N_tx_ant) * (N_tx_ant, N_chip) -> (N_ue_ant, N_chip)
    Y_com_los = H_com_los * X_t_tau_los.';
    % NLoS contribution
    H_mat_nlos = reshape(H_com_nlos, N_ue_ant, N_tx_ant * N_ref_tars);
    X_mat_nlos = reshape(permute(X_t_tau_nlos, [2, 3, 1]), N_ref_tars * N_tx_ant, N_chip);
    Y_com_nlos = H_mat_nlos * X_mat_nlos; % (N_ue_ant, N_chip)

    Y_t_com = Y_com_los + Y_com_nlos;
    Y_t_com = func_add_noise(Y_t_com, Noise_power_com);
    Data_UE(i, :, :) = Y_t_com.';

    if i == N_block
        H_com_los_last = H_com_los;
        H_com_nlos_last = H_com_nlos;
        Delay_los_last = Delay_los;
        Delay_nlos_last = Delay_nlos;

        H_com_los_all(i+1, :,:) = H_com_los;
        H_com_nlos_all(i+1, :, :, :) = H_com_nlos;
    end
end


% Extra symbol N_block+1: delayed tail arriving at UE
X_t_tau_los_last = func_get_x_delay(Data_Tx_pad, N_block+1, Delay_los_last,  T_chip);  % (L_block, N_tx_ant)
X_t_tau_nlos_last = func_get_x_delay(Data_Tx_pad, N_block+1, Delay_nlos_last, T_chip);  % (L_block, N_tx_ant, N_ref_tars)

% LOS contribution
Y_com_los_last = H_com_los_last * X_t_tau_los_last.';
% NLoS contribution
H_mat_nlos_last = reshape(H_com_nlos_last, N_ue_ant,N_tx_ant * N_ref_tars);
X_mat_nlos_last = reshape(permute(X_t_tau_nlos_last, [2, 3, 1]),N_ref_tars * N_tx_ant, N_chip);
Y_com_nlos_last = H_mat_nlos_last * X_mat_nlos_last;

Y_t_com = Y_com_los_last + Y_com_nlos_last;
Y_t_com = func_add_noise(Y_t_com, Noise_power_com);
Data_UE(N_block+1, :, :) = Y_t_com.';


%% Transmit-signal figure (optional) -----------------------------------------
if ~exist('NO_SIGNAL_PLOT','var') || ~NO_SIGNAL_PLOT
% Visualizes Communication, Radar, ISAC signals + Tx autocorrelation.
%
% Subplot 1 : Communication signal (BPSK on even slots | 0 on odd slots)
% Subplot 2 : Radar signal (repeated PRBS / C_w_time)
% Subplot 3 : ISAC signal (odd = PRBS, even = PRBS × BPSK)
% Subplot 4 : Tx waveform autocorrelation per block (dB, overlaid)

%% ── Settings ─────────────────────────────────────────────────────────────────
N_plot  = 2;
N_total = N_plot * N_chip;
t_total = (0:N_total-1) * T_chip * 1e6;   % µs
T_blk_us = N_chip * T_chip * 1e6;

% LaTeX interpreters globally
set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

%% ── Build waveforms ──────────────────────────────────────────────────────────
radar_signal_all = zeros(N_total, 1);
com_even_all     = NaN(N_total, 1);
com_odd_all      = NaN(N_total, 1);
isac_signal_all  = zeros(N_total, 1);
acf_all          = cell(N_plot, 1);   % autocorrelation per block

for i = 1:N_plot
    blk    = (i-1)*N_chip + 1 : i*N_chip;
    bpsk_i = func_bpsk_mod(Data_Tx_bin(i, :, 1));
    tx_sig = squeeze(Data_Tx(i, :, 1)).';   % (N_chip × 1)

    radar_signal_all(blk) = C_w_time;
    isac_signal_all(blk)  = tx_sig;

    sym_idx = 1;
    for slot = 1:N_slot
        idx = blk(1) + (slot-1)*L_slot_per_sym - 1 + (1:L_slot_per_sym);
        if mod(slot, 2) == 0
            com_even_all(idx) = bpsk_i(sym_idx);
            sym_idx = sym_idx + 1;
        else
            com_odd_all(idx) = 0;
        end
    end

    % Normalised autocorrelation of the ISAC Tx waveform for block i
    [acf_i, lags] = xcorr(tx_sig, 'normalized');
    acf_all{i} = 20 * log10(abs(acf_i) + 1e-6);   % dB
end

%% ── Helpers ──────────────────────────────────────────────────────────────────
% Block boundary dividers (dashed vertical lines, shared x = time axis)
draw_block_dividers = @(ax) arrayfun(@(k) ...
    xline(ax, k*T_blk_us, ':', 'Color', [0.6 0.6 0.6], ...
          'LineWidth', 1.6, 'HandleVisibility', 'off'), ...
    1:N_plot-1);

% Per-block rectangle border on the time-domain subplots
draw_block_borders = @(ax) arrayfun(@(k) ...
    rectangle(ax, 'Position', [(k-1)*T_blk_us, -1.5, T_blk_us, 3.0], ...
              'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 1.2), ...
    1:N_plot);

% Block label on top of subplot 1 only
label_blocks = @(ax) arrayfun(@(k) ...
    text(ax, (k-0.5)*T_blk_us, 1.28, sprintf('B%d', k), ...
         'HorizontalAlignment', 'center', 'FontSize', 8, ...
         'Color', [0.2 0.2 0.2], 'FontWeight', 'bold', 'Interpreter', 'latex'), ...
    1:N_plot);

% Distinct colours for the N_plot blocks in subplot 4
block_colors = lines(N_plot);

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Position', [80 60 1400 960], 'Color', 'w');

% ── Subplot 1 : Communication signal ─────────────────────────────────────────
ax1 = subplot(4, 1, 1);
hold on;
h_odd  = stairs(t_total, com_odd_all,  'LineWidth', 1.5, 'Color', [0.65 0.65 0.65]);
h_even = stairs(t_total, com_even_all, 'LineWidth', 1.5, 'Color', [0.00 0.45 0.74]);
draw_block_borders(ax1);
draw_block_dividers(ax1);
label_blocks(ax1);
hold off;
ylim([-1.5 1.5]);  xlim([0 N_plot*T_blk_us]);
grid on;  box on;
title('Communication Signal $\vec{s}_{\rm prbs}$  (Even slots: BPSK $|$ Odd slots: $s=0$)', ...
      'FontWeight', 'bold');
ylabel('Amplitude');
yticks([-1 0 1]);
legend([h_even h_odd], {'Even slots (BPSK)', 'Odd slots (no data, $s=0$)'}, ...
       'Location', 'southoutside', 'Orientation', 'horizontal');

% ── Subplot 2 : Radar signal ──────────────────────────────────────────────────
ax2 = subplot(4, 1, 2);
hold on;
stairs(t_total, radar_signal_all, 'LineWidth', 1.0, 'Color', [0.85 0.33 0.10]);
draw_block_borders(ax2);
draw_block_dividers(ax2);
hold off;
ylim([-1.5 1.5]);  xlim([0 N_plot*T_blk_us]);
grid on;  box on;
title('Radar Signal $\vec{p}_{\rm prbs}$  (Repeated PRBS / $C_w$)', 'FontWeight', 'bold');
ylabel('Amplitude');
yticks([-1 0 1]);

% ── Subplot 3 : ISAC signal ───────────────────────────────────────────────────
ax3 = subplot(4, 1, 3);
hold on;

% Shade even slots (comm-embedded) light blue
for i = 1:N_plot
    blk_t0 = (i-1)*T_blk_us;
    for slot = 1:N_slot
        if mod(slot, 2) == 0
            xs = blk_t0 + (slot-1)*L_slot_per_sym*T_chip*1e6;
            xe = blk_t0 +  slot   *L_slot_per_sym*T_chip*1e6;
            patch([xs xe xe xs], [-1.5 -1.5 1.5 1.5], ...
                  [0.68 0.85 1.0], 'EdgeColor', 'none', ...
                  'FaceAlpha', 0.35, 'HandleVisibility', 'off');
        end
    end
end
stairs(t_total, isac_signal_all, 'LineWidth', 1.0, 'Color', [0.47 0.67 0.19]);
draw_block_borders(ax3);
draw_block_dividers(ax3);
hold off;

ylim([-1.5 1.5]);  xlim([0 N_plot*T_blk_us]);
grid on;  box on;
title('ISAC Signal $\mathbf{D}_{\rm isac}$  (Odd slots: PRBS $|$ Even slots: PRBS $\times$ BPSK)', ...
      'FontWeight', 'bold');
ylabel('Amplitude');
yticks([-1 0 1]);

% Two plot() Line dummies — identical type, so [h1 h2] concatenates safely.
% A filled square marker stands in for the patch shading swatch.
hold on;
h_leg1 = plot(nan, nan, 's', 'MarkerSize', 11, 'LineStyle', 'none', ...
              'MarkerFaceColor', [0.68 0.85 1.0], 'MarkerEdgeColor', 'none');
h_leg2 = plot(nan, nan, '-', 'Color', [0.47 0.67 0.19], 'LineWidth', 2.0);
hold off;
legend([h_leg1 h_leg2], {'Comm-embedded (even) slots', 'ISAC waveform'}, ...
       'Location', 'southoutside', 'Orientation', 'horizontal');

% ── Subplot 4 : Tx autocorrelation per block ──────────────────────────────────
ax4 = subplot(4, 1, 4);
hold on;

leg_handles = gobjects(N_plot, 1);
for i = 1:N_plot
    leg_handles(i) = plot(lags, acf_all{i}, ...
        'LineWidth', 1.4, 'Color', block_colors(i, :));
end

% Highlight the zero-lag peak
xline(ax4, 0, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2, ...
      'HandleVisibility', 'off');

hold off;
xlim([-N_chip  N_chip]);
ylim([-50  5]);
grid on;  box on;
xlabel('Lag (chips)', 'FontSize', 11);
ylabel('Correlation (dB)');
title('Tx Waveform Autocorrelation per Block  (ISAC sequence $\approx$ near-delta $\Rightarrow$ flat sidelobes)', ...
      'FontWeight', 'bold');

leg_labels = arrayfun(@(k) sprintf('Block %d', k), 1:N_plot, 'UniformOutput', false);
legend(leg_handles, leg_labels, ...
       'Location', 'southoutside', 'Orientation', 'horizontal');

%% ── Super-title ──────────────────────────────────────────────────────────────
sgtitle(sprintf('PMCW ISAC Waveform  ---  First %d Blocks  (%d chips/block, $f_c=%.0f$ GHz, $B=%.0f$ MHz)', ...
        N_plot, N_chip, fc/1e9, B/1e6), ...
        'FontWeight', 'bold', 'FontSize', 13, 'Interpreter', 'latex');

%% ── Link time-domain x-axes (subplots 1-3 share time axis) ──────────────────
linkaxes([ax1 ax2 ax3], 'x');
% ax4 has chip-lag x-axis — NOT linked to the others

xlabel(ax3, sprintf('Time ($\\mu$s)  [%d blocks, $T_{\\rm prbs} = %.2f\\,\\mu$s each]', ...
       N_plot, T_blk_us));

fprintf('\nSubplots 1-3 x-axes linked (time). Subplot 4 = chip-lag axis (independent).\n');
end

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function C_w_time = func_generate_waveform(N_chip)
    % generate_waveform generates an Augmented m-sequence of even length.
    % N_chip MUST be a power of 2 (e.g., 64, 128, 256, 512, 1024)
    
    % 1. Validate that N_chip is a power of 2
    n = log2(N_chip);
    if mod(n, 1) ~= 0
        error('For an augmented m-sequence, N_chip must be a power of 2 (e.g., 128, 256).');
    end

    % 2. Define primitive polynomials for the LFSR (Lengths 16 to 4096)
    % The numbers represent the feedback tap positions.
    switch n
        case 4,  taps = [4, 3];             % N = 16
        case 5,  taps = [5, 2];             % N = 32
        case 6,  taps = [6, 1];             % N = 64
        case 7,  taps = [7, 1];             % N = 128
        case 8,  taps = [8, 4, 3, 2];       % N = 256
        case 9,  taps = [9, 4];             % N = 512
        case 10, taps = [10, 3];            % N = 1024
        case 11, taps = [11, 2];            % N = 2048
        case 12, taps = [12, 6, 4, 1];      % N = 4096
        otherwise
            error('Unsupported length. N_chip must be between 16 and 4096.');
    end

    % 3. Initialize the LFSR
    L = N_chip - 1;         % The standard m-sequence length
    state = ones(1, n);     % Initial state of registers (must be non-zero)
    m_seq = zeros(1, L);    % Pre-allocate memory

    % 4. Generate the standard m-sequence (0s and 1s)
    for i = 1:L
        % The output is the last register
        m_seq(i) = state(n);
        
        % Calculate the feedback bit using XOR on the tap positions
        feedback = 0;
        for t = 1:length(taps)
            feedback = xor(feedback, state(taps(t)));
        end
        
        % Shift the registers and insert the feedback bit at the beginning
        state = [feedback, state(1:end-1)];
    end

    % 5. Convert unipolar (0, 1) to bipolar (+1, -1)
    % 0 maps to +1, and 1 maps to -1
    m_seq_bipolar = 1 - 2 * m_seq; 

    % 6. Augment the sequence to make the length even
    % We append a copy of the first chip to the end. This maintains the 
    % circular nature of the sequence for FFT-based correlation processing.
    augmented_seq = [m_seq_bipolar, 1];
    
    % 7. Format as a column vector (standard for time-domain signals in MATLAB)
    C_w_time = augmented_seq(:);
end
function [H_com_los, Range_UE, Radial_UE_vel] = h_com_los(N_ue, UE_position, UE_vel, N_ue_ant, Tx_position, N_tx_ant, P_tx, G_tx, G_ue, Lambda, fc, T_step)
    
    c = physconst('LightSpeed');
    Rel_positions = UE_position - Tx_position; % (3, N_ue)
    [Azi_Tx_UE, Ele_Tx_UE, Range_Tx_UE] = cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_UE = rad2deg(Azi_Tx_UE); % (1, N_ue)
    Ele_Tx_UE = rad2deg(Ele_Tx_UE);
    
    % Convert to spherical coordinates angle at UE
    Rel_positions = Tx_position - UE_position; % (3, N_ue)
    [Azi_UE, Ele_UE, Range_UE] = cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_UE = rad2deg(Azi_UE); % (1, N_ue)
    Ele_UE = rad2deg(Ele_UE);
    
    % Compute the antenna array beetween Tx and UE (downlink)
    Tx_UE_array = cell2mat(arrayfun(@(i) func_get_array_response(N_tx_ant, Lambda/2, Azi_Tx_UE(i), 90-Ele_Tx_UE(i), Lambda, Tx_position), 1:N_ue, 'UniformOutput', false));
    UE_array = cell2mat(arrayfun(@(i) func_get_array_response(N_ue_ant, Lambda/2, Azi_UE(i), 90-Ele_UE(i), Lambda, UE_position, false), 1:N_ue, 'UniformOutput', false));
    

    % Compute radial velocities
    Rel_positions = UE_position - Tx_position; % (3, N_ue)
    R_hat = Rel_positions / norm(Rel_positions);   % unit vector
    Radial_UE_vel = dot(UE_vel, R_hat);   % radial velocity
    
    P_tx_ant = (P_tx / N_tx_ant) * ones(N_tx_ant,1); % Tx power of each antenna P_tx / N_tx_ant with shape N_tx_ant x 1 
    G_tx_ant = G_tx * ones(N_tx_ant,1); % Tx gain: N_tx_ant x 1
    G_ue_ant = G_ue * ones(N_ue_ant,1); % Tx gain: N_tx_ant x 1 
    A_e_ue_ant = G_ue_ant * Lambda^2 / (4 * pi); % Effective antenna aperture 
    PL_com_los = 1 / (4 * pi * Range_Tx_UE.^2); % 1 x N_ue
    Delay_com_los = Range_Tx_UE / c;  % 1 x N_ue
    Doppler_com_los = -Radial_UE_vel / Lambda;  % 1 x N_ue
     
    H_com_los = (sqrt(PL_com_los) .* UE_array .* sqrt(A_e_ue_ant)) ...        % N_ue_ant × 1
              * (Tx_UE_array .* sqrt(P_tx_ant).* sqrt(G_tx_ant)).' ...               % 1 × N_tx_ant
              * exp(-1j * 2*pi * fc * Delay_com_los) ...           % carrier phase
              * exp( 1j * 2*pi * Doppler_com_los * T_step);       % Doppler phase
end
function [H_com_nlos, Range_Tx_ref_tars, Range_ref_tars_UE, Radial_ref_tars_vel] = h_com_nlos(N_ref_tars, All_tars_position, All_radial_tars_vel, All_radial_tars_rcs, Tx_position, N_tx_ant, UE_position, N_ue_ant, Radial_UE_vel, P_tx, G_tx, G_ue, Lambda, fc,T_step)
    
    c = physconst('LightSpeed');
    
    Rel_positions  = UE_position - All_tars_position;                % 3 × (N_scats + N_tars)
    Range_All_tars = sqrt(sum(Rel_positions.^2, 1));
    [sorted_ranges, idx] = sort(Range_All_tars, 'ascend');
    closest_idx = idx(1:N_ref_tars);
    
    % Compute atenna array Tx -> ref tars
    Ref_tars_position = All_tars_position(:, closest_idx);
    Rel_positions  = Ref_tars_position - Tx_position;                % 3 × N_ref_tars
    [Azi_Tx_ref_tars, Ele_Tx_ref_tars, Range_Tx_ref_tars] = cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_ref_tars = rad2deg(Azi_Tx_ref_tars); % (1, N_ref_tars)
    Ele_Tx_ref_tars = rad2deg(Ele_Tx_ref_tars);
    Tx_Ref_tars_array = cell2mat(arrayfun(@(i) func_get_array_response( ...
        N_tx_ant, Lambda/2, ...
        Azi_Tx_ref_tars(i), 90 - Ele_Tx_ref_tars(i), ...
        Lambda, Tx_position), ...
        1:N_ref_tars, 'UniformOutput', false));
    
    % Compute atenna array ref tars -> UE
    Rel_positions  = UE_position - Ref_tars_position;   % 3 × N_ref_tars
    [Azi_ref_tars_UE, Ele_ref_tars_UE, Range_ref_tars_UE] = cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_ref_tars_UE = rad2deg(Azi_ref_tars_UE); % (1, N_ref_tars)
    Ele_ref_tars_UE = rad2deg(Ele_ref_tars_UE);
    Ref_tars_UE_array = cell2mat(arrayfun(@(i) func_get_array_response( ...
        N_ue_ant, Lambda/2, ...
        Azi_ref_tars_UE(i), 90 - Ele_ref_tars_UE(i), ...
        Lambda, UE_position, false), ...
        1:N_ref_tars, 'UniformOutput', false));
    
    % Get path loss, delay, doppler of the ref tars
    Radial_ref_tars_vel = All_radial_tars_vel(:, closest_idx);
    PL_nlos_comm = 1 ./ ((4 * pi * Range_Tx_ref_tars.^2) .* (4 * pi * Range_ref_tars_UE.^2));  % 1 × N_ref_tars
    Delay_nlos_comm = (Range_Tx_ref_tars + Range_ref_tars_UE) / c;            % 1 × N_ref_tars
    Doppler_nlos_comm  = - (Radial_UE_vel + Radial_ref_tars_vel) / Lambda;              % 1 × N_ref_tars
    
    Ref_tars_rcs = All_radial_tars_rcs(:, closest_idx); % 1 x N_ref_tar
    
    P_tx_ant = (P_tx / N_tx_ant) * ones(N_tx_ant,1); % Tx power of each antenna P_tx / N_tx_ant with shape N_tx_ant x 1 
    G_tx_ant = G_tx * ones(N_tx_ant,1); % Tx gain: N_tx_ant x 1
    G_ue_ant = G_ue * ones(N_ue_ant,1); % Tx gain: N_tx_ant x 1
    A_e_ue_ant = G_ue_ant * Lambda^2 / (4 * pi); % Effective antenna aperture 
    H_com_nlos = zeros(N_ue_ant, N_tx_ant, N_ref_tars);
     
    for i = 1:N_ref_tars
        H_com_nlos(:,:,i) = sqrt(Ref_tars_rcs(i)) ...                         % scatter reflectance amplitude
            * (sqrt(PL_nlos_comm(i)) .* Ref_tars_UE_array(:,i) .* sqrt(A_e_ue_ant)) ...   % UE Rx side  (N_ue_ant × 1)
            * (Tx_Ref_tars_array(:,i) .* sqrt(P_tx_ant).* sqrt(G_tx_ant)).' ...                 % Tx side     (1 × N_tx_ant)
            * exp(-1j * 2*pi * fc * Delay_nlos_comm(i)) ...                 % carrier phase shift
            * exp( 1j * 2*pi * Doppler_nlos_comm(i) * T_step);            % Doppler phase shift
    end
end
function y = func_add_noise(x, noise_power)

    % Generate complex AWGN with correct power
    noise = sqrt(noise_power/2) * (randn(size(x)) + 1j * randn(size(x)));
    
    % Add noise
    y = x + noise;

end