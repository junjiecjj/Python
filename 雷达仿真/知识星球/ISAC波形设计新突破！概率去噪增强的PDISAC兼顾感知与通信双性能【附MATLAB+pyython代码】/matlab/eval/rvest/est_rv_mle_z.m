function [hat_r_tars, hat_v_tars] = est_rv_mle_z(z_time, p_time, noise_power, P_sensing, target_distance, target_vel, fc, T_fast, T_slow)

    c = physconst('LightSpeed');
    [N_fast, N_slow] = size(z_time);

    p_time = p_time(:);
    if length(p_time) ~= N_fast
        error('PRBS length (%d) must match fast-time dimension (%d)', length(p_time), N_fast);
    end

    % --- Precompute periodic autocorrelation of the PRBS (once) ---
    p_freq  = fft(p_time);                 % N_fast x 1
    R_p = ifft(p_freq .* conj(p_freq));        % N_fast x 1, real-valued ideally

    % --- Model amplitude ---
    A  = sqrt(P_sensing);
    r0 = target_distance;
    v0 = target_vel;

    % --- Build NLL over z-domain using precomputed R_p ---
    nll_func = @(theta) compute_nll( ...
        theta, z_time, R_p, A, noise_power, T_fast, T_slow, fc, c, N_fast, N_slow);

    options = optimset('Display', 'off', 'MaxIter', 2000, 'MaxFunEvals', 5000, ...
                       'TolX', 1e-8, 'TolFun', 1e-8);

    [theta_opt, ~] = fminsearch(nll_func, [r0; v0], options);

    hat_r_tars = theta_opt(1);
    hat_v_tars = theta_opt(2);
end



function nll = compute_nll(theta, Z_obs, R_p, A, noise_power, ...
                           T_fast, T_slow, fc, c, N_fast, N_slow)
    r = theta(1);
    v = theta(2);

    % Model matched-filtered output directly
    Z_model = generate_signal_model_parametric( ...
                  r, v, A, R_p, T_fast, T_slow, fc, c, N_fast, N_slow);

    residual = Z_obs - Z_model;
    nll = sum(abs(residual(:)).^2) / (2 * noise_power);
end



function Z_model = generate_signal_model_parametric(r, v, A, R_p, ...
    T_fast, T_slow, fc, c, N_fast, N_slow)
    % Matched-filter domain model:
    % Z[:,m] = A * exp(j*(2*pi*fd*m*T_slow - phi0)) * circshift(R_p, n0)
    %
    % Inputs:
    %   r, v     - range, velocity
    %   A        - amplitude (sqrt(received power))
    %   R_p      - periodic autocorrelation of PRBS (ifft(fft(p).*conj(fft(p))))
    %   T_fast   - chip duration (s)
    %   T_slow - slow-time spacing (bit period) (s)
    %   fc, c    - carrier frequency (Hz), speed of light (m/s)
    %   N_fast   - fast-time length (chips)
    %   N_slow   - number of bits (columns)

    % --- Delay in chips (integer model, consistent with your code) ---
    tau = 2 * r / c;                  % round-trip delay (s)
    n0  = round(tau / T_fast);        % integer chip delay
    n0_wrapped = mod(n0, N_fast);

    % --- Doppler (keep your sign convention) and carrier phase ---
    fd   = -2 * v * fc / c;           % Hz
    phi0 = 4 * pi * fc * r / c;       % rad

    % --- Slow-time phase evolution over bit period ---
    m = 0:(N_slow-1);
    phase_total = 2*pi*fd*m*T_slow - phi0; % 1 x N_slow
    exp_phase   = exp(1j * phase_total);     % 1 x N_slow

    % --- Build matched-filtered columns via circular shift of R_p ---
    % Z[:,m] = A * exp_phase[m] * circshift(R_p, n0)
    Z_model = A * (circshift(R_p(:), n0_wrapped)) * exp_phase; % N_fast x N_slow
end