function [est_R, est_v] = func_mle_estimation( ...
        Data_Rx, Data_Tx, N_tars, ...
        Tars_position, Tars_rcs, Tars_vel, ...
        Tx_position, N_tx_ant, Rx_position, N_rx_ant, ...
        P_tx, G_tx, G_rx, Lambda, fc, Noise_power_sen, ...
        T_chip, N_chip, N_block)
%FUNC_MLE_ESTIMATION  MLE of range and radial velocity per target.
%
% APPROACH — SINGLE-TARGET DATA RECONSTRUCTION
% ─────────────────────────────────────────────
%  The received signal is a linear sum of all target contributions:
%
%    Data_Rx = Y_1 + Y_2 + ... + Y_{N_tars} + noise
%
%  For each target m we isolate its single-target signal by removing
%  all other contributions using the KNOWN INITIAL GEOMETRY:
%
%    Data_Rx_m = Data_Rx − Σ_{k≠m} Y_k( Range_tars(k), Radial_tars_vel(k) )
%
%  Key points:
%    • Range_tars and Radial_tars_vel are computed directly from the given
%      initial positions — they are NOT MLE estimates, so they carry no
%      estimation error that could propagate.
%    • The subtracted Y_k uses exactly the same generate_signal_model that
%      was used to build the simulation, so the cancellation is accurate.
%    • Data_Rx_m is then a single-target observation corrupted only by
%      noise + residual model mismatch (moving target range walk), which
%      is small over short CPI durations.
%    • The MLE for target m runs on Data_Rx_m alone → unbiased, no
%      inter-target interference regardless of N_tars or P_tx.
%    • No iterative passes are needed.
%
% SIGNAL MODEL (perfect calibration assumed)
% ──────────────────────────────────────────
%  h_sen_tars at block b (motion model advances T_pmcw BEFORE the call):
%
%    H_b(:,:,m) = sqrt(RCS_m)
%        * ( sqrt(PL_m) .* Rx_array_m .* sqrt(A_e) )
%        * ( Tx_array_m .* sqrt(P_ant) .* sqrt(G_tx) ).'
%        * exp(-j2π fc 2R_0/c)           ← carrier phase at initial range
%        * exp( j2π Doppler_m b T_pmcw)  ← inter-block Doppler (b-based, 1..N)
%
%  generate_signal_model reconstructs Y_m = X_delayed_b * H_b.' per block.
%
%  Phase convention:
%    A_R * phase_b = sqrt(PL) * exp(-j2π fc 2R_0/c)
%                             * exp(-j2π fc (2v/c) b T_pmcw)
%                 = sqrt(PL) * exp(-j2π fc 2(R_0 + v*b*T_pmcw)/c)
%                 = sqrt(PL) * exp(-j2π fc 2 R_b/c)   ✓ matches h_sen_tars
%
%  squeeze() on func_get_x_delay output: the function always returns
%  (N_chip, N_tx, N_tars); for scalar delay N_tars=1 → (N_chip, N_tx, 1).
%  squeeze() → (N_chip, N_tx) so X_b * H_b.' is a valid matrix multiply.
%
% Inputs
%   Data_Rx         (N_block × N_chip × N_rx_ant)
%   Data_Tx         (N_block × N_chip × N_tx_ant)
%   N_tars          number of targets
%   Tars_position   (3 × N_tars)  initial positions [m]
%   Tars_rcs        (1 × N_tars)  radar cross-sections [m²]
%   Tars_vel        (3 × N_tars)  velocity vectors [m/s]
%   Tx_position     (3 × 1)
%   N_tx_ant        number of Tx antennas
%   Rx_position     (3 × 1)
%   N_rx_ant        number of Rx antennas
%   P_tx            total transmit power [W]
%   G_tx / G_rx     antenna gains [linear]
%   Lambda          wavelength [m]
%   fc              carrier frequency [Hz]
%   Noise_power_sen noise power σ² [W]
%   T_chip          chip duration [s]
%   N_chip          chips per block
%   N_block         number of blocks
%
% Outputs
%   est_R  (N_tars × 1)  MLE range estimates [m]
%   est_v  (N_tars × 1)  MLE radial velocity estimates [m/s]

    c      = physconst('LightSpeed');
    T_pmcw = N_chip * T_chip;

    %% ── Geometry ─────────────────────────────────────────────────────────────
    Rel_positions = Tars_position - Tx_position;   % (3 × N_tars)

    [Azi_Tx_tars, Ele_Tx_tars, Range_tars] = cart2sph( ...
        Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_tars = rad2deg(Azi_Tx_tars);
    Ele_Tx_tars = rad2deg(Ele_Tx_tars);

    Azi_tars_Rx = Azi_Tx_tars;   % monostatic
    Ele_tars_Rx = Ele_Tx_tars;

    Tx_tars_array = cell2mat(arrayfun(@(i) func_get_array_response( ...
        N_tx_ant, Lambda/2, Azi_Tx_tars(i), 90-Ele_Tx_tars(i), Lambda, Tx_position), ...
        1:N_tars, 'UniformOutput', false));   % (N_tx_ant × N_tars)

    Tars_Rx_array = cell2mat(arrayfun(@(i) func_get_array_response( ...
        N_rx_ant, N_tx_ant*Lambda/2, Azi_tars_Rx(i), 90-Ele_tars_Rx(i), Lambda, Rx_position), ...
        1:N_tars, 'UniformOutput', false));   % (N_rx_ant × N_tars)

    P_tx_ant   = (P_tx / N_tx_ant) * ones(N_tx_ant, 1);
    G_tx_ant   = G_tx              * ones(N_tx_ant, 1);
    G_rx_ant   = G_rx              * ones(N_rx_ant, 1);
    A_e_rx_ant = G_rx_ant * Lambda^2 / (4*pi);

    % Radial velocity per target from initial positions
    Radial_tars_vel = zeros(1, N_tars);
    for m = 1:N_tars
        R_hat              = Rel_positions(:,m) / norm(Rel_positions(:,m));
        Radial_tars_vel(m) = dot(Tars_vel(:,m), R_hat);
    end

    % Spatial steering matrix per target — angle structure only, no PL/phase
    Alpha = cell(N_tars, 1);
    for m = 1:N_tars
        Alpha{m} = sqrt(Tars_rcs(m)) * ...
            (Tars_Rx_array(:,m) .* sqrt(A_e_rx_ant)) * ...
            (Tx_tars_array(:,m) .* sqrt(P_tx_ant) .* sqrt(G_tx_ant)).';  % (N_rx_ant × N_tx_ant)
    end

    %% ── Pre-compute each target's full signal model at initial geometry ───────
    %
    %  Y_init{k} is the model signal of target k evaluated at the geometry-
    %  derived (Range_tars(k), Radial_tars_vel(k)).  These are used to cancel
    %  other targets from Data_Rx before running MLE on target m.
    %
    %  Because Range_tars / Radial_tars_vel come from the given initial positions
    %  (identical to what h_sen_tars used at block 1 of the simulation), the
    %  cancellation is accurate and does not depend on any prior MLE estimates.
    %
    Y_init = cell(N_tars, 1);
    for k = 1:N_tars
        Y_init{k} = generate_signal_model( ...
            Range_tars(k), Radial_tars_vel(k), Data_Tx, Alpha{k}, ...
            T_chip, T_pmcw, fc, Lambda, c, N_chip, N_block, N_tx_ant, N_rx_ant);
    end

    %% ── fminsearch options ───────────────────────────────────────────────────
    options = optimset('Display',     'off', ...
                       'MaxIter',     2000,  ...
                       'MaxFunEvals', 5000,  ...
                       'TolX',        1e-8,  ...
                       'TolFun',      1e-8);

    %% ── MLE per target on isolated single-target signal ─────────────────────
    est_R = zeros(N_tars, 1);
    est_v = zeros(N_tars, 1);

    for m = 1:N_tars

        %% Reconstruct Data_Rx_m: remove all targets EXCEPT m -----------------
        %
        %  Data_Rx   = Y_1 + Y_2 + ... + Y_N + noise
        %  Data_Rx_m = Data_Rx − Σ_{k≠m} Y_init{k}
        %            ≈ Y_m + noise       (single-target observation)
        %
        %  The MLE cost function now sees only target m's signal plus noise,
        %  so the NLL minimum is unbiased regardless of N_tars or P_tx.
        %
        Data_Rx_m = Data_Rx;
        for k = 1:N_tars
            if k ~= m
                Data_Rx_m = Data_Rx_m - Y_init{k};
            end
        end

        %% NLL on single-target observation -----------------------------------
        nll_func = @(theta) compute_nll( ...
            theta, Data_Rx_m, Data_Tx, Alpha{m}, ...
            Noise_power_sen, T_chip, T_pmcw, fc, Lambda, c, ...
            N_chip, N_block, N_tx_ant, N_rx_ant);

        theta_opt = fminsearch(nll_func, [Range_tars(m); Radial_tars_vel(m)], options);

        est_R(m) = theta_opt(1);
        est_v(m) = theta_opt(2);

    end   % for m = 1:N_tars
end


%% ═══════════════════════════════════════════════════════════════════════════
%  LOCAL: compute_nll
%
%  NLL(θ) = (1/2σ²) ‖ Data_Rx_m − Y_model(R,v) ‖²_F
%
%  Data_Rx_m is already a single-target observation (all other targets
%  removed), so this is a pure single-target NLL.
% ═══════════════════════════════════════════════════════════════════════════
function nll = compute_nll(theta, Data_Rx_m, Data_Tx, alpha_m, ...
        Noise_power_sen, T_chip, T_pmcw, fc, Lambda, c, ...
        N_chip, N_block, N_tx_ant, N_rx_ant)

    R = theta(1);
    v = theta(2);

    if R <= 0
        nll = inf;
        return;
    end

    Y_model  = generate_signal_model(R, v, Data_Tx, alpha_m, ...
        T_chip, T_pmcw, fc, Lambda, c, N_chip, N_block, N_tx_ant, N_rx_ant);

    residual = Data_Rx_m - Y_model;
    nll      = sum(abs(residual(:)).^2) / (2 * Noise_power_sen);
end


%% ═══════════════════════════════════════════════════════════════════════════
%  LOCAL: generate_signal_model
%
%  Reconstructs (N_block × N_chip × N_rx_ant) contribution of ONE target.
%
%  At block b (1-based, consistent with the block motion model in main3_signal_channel_model):
%    Y(b,:,:) = X_b * H_b.'
%    H_b      = A_R * phase_b * alpha_m           (N_rx_ant × N_tx_ant)
%    A_R      = sqrt(PL) * exp(-j2π fc Delay)     amplitude + carrier ref
%    phase_b  = exp(j2π Doppler * b * T_pmcw)     inter-block Doppler, b-based
%
%  squeeze() is required: func_get_x_delay returns (N_chip, N_tx, N_tars);
%  scalar delay → N_tars=1 → (N_chip, N_tx, 1); squeeze → (N_chip, N_tx).
% ═══════════════════════════════════════════════════════════════════════════
function Y_model = generate_signal_model(R, v, Data_Tx, alpha_m, ...
        T_chip, T_pmcw, fc, Lambda, c, N_chip, N_block, N_tx_ant, N_rx_ant)

    Delay_m   = 2 * R / c;
    Doppler_m = -2 * v / Lambda;
    PL_m      = 1 / (4*pi*R^2)^2;
    A_R       = sqrt(PL_m) * exp(-1j * 2*pi * fc * Delay_m);

    Y_model = zeros(N_block, N_chip, N_rx_ant);

    for b = 1:N_block
        % squeeze: (N_chip, N_tx, 1) → (N_chip, N_tx)
        X_b = squeeze( ...
            func_get_x_delay(Data_Tx, b, Delay_m, T_chip));   % (N_chip × N_tx_ant)

        % b-based: motion model advances T_pmcw before h_sen_tars call at block b
        phase_b = exp(1j * 2*pi * Doppler_m * b * T_pmcw);

        H_b = A_R * phase_b * alpha_m;     % (N_rx_ant × N_tx_ant)

        Y_model(b, :, :) = X_b * H_b.';   % (N_chip × N_rx_ant)
    end
end