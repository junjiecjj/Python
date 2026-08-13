% COMM_PROCESS_C  (variant of Group 6: communication processing)
% -------------------------------------------------------------------------
% Builds on main6b_comm_process.m: keeps the circular-correlation (ACF)
% UE delay search func_get_delay_index_acf introduced there (unchanged).
% On top of that, the channel-estimation and data-detection steps are
% rewritten to follow the attached, updated 5_2_channel_est.tex and
% 5_3_data_detection.tex exactly:
%
%   Sec. 5.2 (channel estimation) — scalar LS estimate per slot, not a
%   chipwise vector:
%       hat_h_com,i^(k) = (p_odd^H y_p) / (p_odd^H p_odd)          [eq_est_h_com_ik]
%
%   Sec. 5.3 (data detection) — matched-filter statistic z, then
%   equalize by the SCALAR channel estimate times the pilot/data code
%   energy:
%       z_com,i^(k)  = p_even^H y_d                                 [above eq_est_s]
%       hat_s_k,i    = z_com,i^(k) / ( hat_h_com,i^(k) * (p_even^H p_even) )   [eq_est_s]
%
% main6_comm_process.m computed a chipwise vector "h_est = conj(x_p).*y_p"
% and folded the normalization into the matched-filter denominator
% implicitly (which is numerically equivalent here only because the pilot
% and data slots have equal length and unit-modulus chips). This file
% instead computes the scalar channel estimate explicitly, as in the
% manuscript, which is the mathematically correct LS estimator and stays
% correct even if that equal-length/unit-modulus coincidence did not hold.
%
% Nothing else changes: same four CSI/delay conditions, same delay
% synchronisation (func_get_delay_index_acf, from main6b), same
% func_get_perfect_csi, same plotting.
% Prerequisite: main3_signal_channel_model has produced Data_UE, H_com_*_all.

%% Detect --------------------------------------------------------------------

% Construct a buffer from two consecutive blocks to handle propagation delay
Data_UE_buffer = [Data_UE(1:N_block, :), Data_UE(2:(N_block+1), :)];

% Estimate the starting index of the frame using the reference PRBS sequence
% (circular-correlation / ACF version, inherited from main6b)
Index_delay = func_get_delay_index_acf(Data_UE_buffer(:, 1:N_chip), C_w_time, N_slot, L_slot_per_sym);

% Remove delay and extract synchronized chips
Rel_positions = Tx_position - UE_position; % (3, N_ue)
[~, ~, Current_Range_UE] = cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));

Data_UE_no_delay_perf = zeros(N_block, N_chip);
Data_UE_no_delay_est = zeros(N_block, N_chip);
for i = 1:N_block
    d = Index_delay(i);
    Data_UE_no_delay_est(i, :) = Data_UE_buffer(i, d:(d+N_chip-1));

    d = floor(Current_Range_UE/(c * T_chip)) + 1;
    Data_UE_no_delay_perf(i, :) = Data_UE_buffer(i, d:(d+N_chip-1));
end


true_delay_index = floor(Current_Range_UE/(c * T_chip)) + 1;
[H_com_los_all_perf, H_com_nlos_all_perf] = func_get_perfect_csi(H_com_los_all, H_com_nlos_all, N_block, N_chip, N_ref_tars, true_delay_index);


% Transpose to [N_chip x N_block] for column-wise processing
Data_UE_no_delay_est = Data_UE_no_delay_est.';
Data_UE_no_delay_perf = Data_UE_no_delay_perf.';


Data_UE_est_tau_est_h_bpsk = zeros(N_block, N_sym_per_block);
Data_UE_est_tau_perf_h_bpsk = zeros(N_block, N_sym_per_block);
Data_UE_perf_tau_est_h_bpsk = zeros(N_block, N_sym_per_block);
Data_UE_perf_tau_perf_h_bpsk = zeros(N_block, N_sym_per_block);

for b_idx = 1:N_block
    Data_UE_no_delay_est_i = Data_UE_no_delay_est(:, b_idx);
    Data_UE_no_delay_perf_i = Data_UE_no_delay_perf(:, b_idx);
    H_com_los_all_perf_i = H_com_los_all_perf(b_idx, :).';

    for s_idx = 1:N_sym_per_block

        % -----------------------------------
        % Estimated Delay + Estimated Channel
        % -----------------------------------

        % Pilot index (odd slot k) -> p_{prbs,k}^{odd}, y_{com,i}^{odd,(k)}
        p_idx_num = (s_idx-1)*2 + 1;
        idx_p = (p_idx_num-1)*L_slot_per_sym + 1 : p_idx_num*L_slot_per_sym;
        x_p = C_w_time(idx_p);                    % p_{prbs,k}^{odd}  [L_slot_per_sym x 1]
        y_p = Data_UE_no_delay_est_i(idx_p);       % y_{com,i}^{odd,(k)}

        % Scalar LS channel estimate, (eq_est_h_com_ik):
        %   hat_h_{com,i}^{(k)} = (p_odd^H y_p) / (p_odd^H p_odd)
        h_hat = (x_p' * y_p) / (x_p' * x_p);       % scalar

        % Data index (even slot k) -> p_{prbs,k}^{even}, y_{com,i}^{even,(k)}
        d_idx_num = p_idx_num + 1;
        idx_d = (d_idx_num-1)*L_slot_per_sym + 1 : d_idx_num*L_slot_per_sym;
        x_d = C_w_time(idx_d);                     % p_{prbs,k}^{even}  [L_slot_per_sym x 1]
        y_d = Data_UE_no_delay_est_i(idx_d);        % y_{com,i}^{even,(k)}

        % Matched-filter statistic z_{com,i}^{(k)} = p_even^H y_d
        z_com = x_d' * y_d;

        % Equalization, (eq_est_s):
        %   hat_s_{k,i} = z_{com,i}^{(k)} / ( hat_h_{com,i}^{(k)} * (p_even^H p_even) )
        s_est = z_com / (h_hat * (x_d' * x_d));

        % Store BPSK symbol
        Data_UE_est_tau_est_h_bpsk(b_idx, s_idx) = s_est;

        % -----------------------------
        % Estimate Delay + True Channel
        % -----------------------------

        % Data index
        d_idx_num = p_idx_num + 1;
        idx_d = (d_idx_num-1)*L_slot_per_sym + 1 : d_idx_num*L_slot_per_sym;
        x_d = C_w_time(idx_d);
        y_d = Data_UE_no_delay_est_i(idx_d);

        % True channel is block-constant across chips; extract the scalar
        % h_{com,i}^{los} (H_com_los_all_perf_i(idx_d) repeats the same value).
        h_true = mean(H_com_los_all_perf_i(idx_d));

        % Matched-filter statistic
        z_com = x_d' * y_d;

        % Equalization with the known scalar channel, (eq_est_s) form:
        s_perf = z_com / (h_true * (x_d' * x_d));

        % Store BPSK symbol
        Data_UE_est_tau_perf_h_bpsk(b_idx, s_idx) = s_perf;


        % -------------------------------
        % True Delay + Estimated Channel
        % -------------------------------

        % Pilot index (odd slot k)
        p_idx_num = (s_idx-1)*2 + 1;
        idx_p = (p_idx_num-1)*L_slot_per_sym + 1 : p_idx_num*L_slot_per_sym;
        x_p = C_w_time(idx_p);
        y_p = Data_UE_no_delay_perf_i(idx_p);

        % Scalar LS channel estimate, (eq_est_h_com_ik)
        h_hat = (x_p' * y_p) / (x_p' * x_p);

        % Data index (even slot k)
        d_idx_num = p_idx_num + 1;
        idx_d = (d_idx_num-1)*L_slot_per_sym + 1 : d_idx_num*L_slot_per_sym;
        x_d = C_w_time(idx_d);
        y_d = Data_UE_no_delay_perf_i(idx_d);

        % Matched-filter statistic
        z_com = x_d' * y_d;

        % Equalization, (eq_est_s)
        s_est = z_com / (h_hat * (x_d' * x_d));

        % Store BPSK symbol
        Data_UE_perf_tau_est_h_bpsk(b_idx, s_idx) = s_est;



        % --------------------------
        % True Delay + True Channel
        % --------------------------

        % Data index
        d_idx_num = p_idx_num + 1;
        idx_d = (d_idx_num-1)*L_slot_per_sym + 1 : d_idx_num*L_slot_per_sym;
        x_d = C_w_time(idx_d);
        y_d = Data_UE_no_delay_perf_i(idx_d);
        h_true = mean(H_com_los_all_perf_i(idx_d));

        % Matched-filter statistic
        z_com = x_d' * y_d;

        % Equalization with the known scalar channel
        s_perf = z_com / (h_true * (x_d' * x_d));

        % Store BPSK symbol
        Data_UE_perf_tau_perf_h_bpsk(b_idx, s_idx) = s_perf;
    end
end


Data_UE_est_tau_est_h_bin = func_bpsk_demod(Data_UE_est_tau_est_h_bpsk(:));
Data_UE_est_tau_perf_h_bin = func_bpsk_demod(Data_UE_est_tau_perf_h_bpsk(:));
Data_UE_perf_tau_est_h_bin = func_bpsk_demod(Data_UE_perf_tau_est_h_bpsk(:));
Data_UE_perf_tau_perf_h_bin = func_bpsk_demod(Data_UE_perf_tau_perf_h_bpsk(:));



%% BPSK constellation figure (optional) --------------------------------------
if ~exist('NO_BPSK_PLOT','var') || ~NO_BPSK_PLOT

% Constellation (realistic case: estimated delay + estimated channel)
Data_UE_no_delay_bpsk = Data_UE_est_tau_est_h_bpsk;

refconst = func_bpsk_mod([0 1]);
constellationDiagram = comm.ConstellationDiagram('ReferenceConstellation', refconst, 'Title', 'Slot-Pairing BPSK Constellation (Circular-ACF Delay Sync + Scalar LS Channel Est.)');
constellationDiagram(Data_UE_no_delay_bpsk(:));

Data_UE_bin = func_bpsk_demod(Data_UE_no_delay_bpsk(:));
[numErr, bitRate] = biterr(Data_Tx_bin(:), Data_UE_bin(:));


end

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function delay_idx = func_get_delay_index_acf(Data_UE, C_w_time, N_slot, L_slot)
    % FUNC_GET_DELAY_INDEX_ACF  Circular-correlation (ACF) delay search,
    % inherited unchanged from main6b_comm_process.m. Same odd/even slot
    % masking and combine-by-magnitude logic as the original linear-xcorr
    % version; the correlation kernel is the length-N_chip circular
    % correlation via FFT,
    %   R[k] = ifft( fft(y) .* conj(fft(x)) )(k),
    % i.e., the \circledast operator of (eq_acf_com_delay), matching
    % the matched-filter convention already used for sensing in
    % main4_sensing_process.m (Rx_freq .* conj(Tx_freq) then ifft).
    N_block = size(Data_UE, 1);
    N_chip  = size(Data_UE, 2);
    delay_idx = zeros(1, N_block);

    % Build odd/even masks
    mask_odd  = zeros(N_chip, 1);
    mask_even = zeros(N_chip, 1);
    for slot = 1:N_slot
        slot_idx = (slot-1)*L_slot + 1 : slot*L_slot;  % use L_slot
        if mod(slot, 2) ~= 0  % odd
            mask_odd(slot_idx) = 1;
        else                   % even
            mask_even(slot_idx) = 1;
        end
    end

    C_odd  = C_w_time .* mask_odd;   % reference for odd slots
    C_even = C_w_time .* mask_even;  % reference for even slots

    % Frequency-domain references, computed once and reused for every block
    C_odd_freq  = conj(fft(C_odd));
    C_even_freq = conj(fft(C_even));

    for i = 1:N_block
        y_i    = Data_UE(i, :).';     % N_chip x 1
        Y_freq = fft(y_i);

        % Odd slots: coherent (known, no data) -- circular correlation
        R_odd  = ifft(Y_freq .* C_odd_freq);    % N_chip x 1, lag k = 1..N_chip

        % Even slots: non-coherent (unknown because of data) -- circular correlation
        R_even = ifft(Y_freq .* C_even_freq);

        % Combine: coherent odd + non-coherent even
        R_total = abs(R_odd) + abs(R_even);

        [~, peak] = max(R_total);
        delay_idx(i) = peak;
    end
end

function [H_com_los_all_perf, H_com_nlos_all_perf] = func_get_perfect_csi(H_com_los_all, H_com_nlos_all, N_block, N_chip, N_ref_tars, delay_index)
    H_com_los_all_buffer  = repmat(squeeze(H_com_los_all),  1, N_chip);
    H_com_los_all_buffer = [H_com_los_all_buffer(1:N_block, :), H_com_los_all_buffer(2:(N_block+1), :)];
    H_com_nlos_all_buffer = repmat(squeeze(H_com_nlos_all), 1, 1, N_chip);
    H_com_nlos_all_buffer = cat(3, H_com_nlos_all_buffer(1:N_block, :, :), H_com_nlos_all_buffer(2:(N_block+1), :, :));
    H_com_los_all_perf = zeros(N_block, N_chip);
    H_com_nlos_all_perf = zeros(N_block, N_ref_tars, N_chip);
    for i = 1:N_block
        H_com_los_all_perf(i, :) = H_com_los_all_buffer(i, delay_index:(delay_index+N_chip-1));
        H_com_nlos_all_perf(i, :, :) = H_com_nlos_all_buffer(i, :, delay_index:(delay_index+N_chip-1));
    end
end
