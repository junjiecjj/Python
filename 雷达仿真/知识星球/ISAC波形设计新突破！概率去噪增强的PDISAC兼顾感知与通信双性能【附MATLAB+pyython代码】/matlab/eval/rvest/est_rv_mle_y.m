function [hat_r_tars, hat_v_tars] = est_rv_mle_y(y_time, p_time, noise_power, P_sensing, target_distance, target_vel, fc, T_fast, T_slow)
    % MLE estimator for PMCW radar following the exact signal model:
    % y[n,m] = A * p[n-n0] * exp(j*(2*pi*fd*m*T_slow - phi0)) + w[n,m]
    
    % Physical constants
    c = physconst('LightSpeed');
    
    % Get dimensions
    [N_fast, N_slow] = size(y_time);
    
    % Ensure p_time is column vector
    p_time = p_time(:);
    
    if length(p_time) ~= N_fast
        error('PRBS length (%d) must match fast-time dimension (%d)', length(p_time), N_fast);
    end
    
    % Signal amplitude from received power
    A = sqrt(P_sensing);
    
    % Initial guess
    r0 = target_distance;
    v0 = target_vel;
    
    
    % Negative log-likelihood function
    nll_func = @(theta) compute_nll(theta, y_time, p_time, A, ...
                                    noise_power, T_fast, T_slow, fc, c, N_fast, N_slow);
    
    % Initial guess vector [range; velocity]
    theta0 = [r0; v0];
    
    % Optimization options for fminsearch
    options = optimset('Display', 'off', ...
                      'MaxIter', 2000, ...
                      'MaxFunEvals', 5000, ...
                      'TolX', 1e-8, ...
                      'TolFun', 1e-8);
    
    % Run optimization using fminsearch (Nelder-Mead simplex)
    [theta_opt, fval] = fminsearch(nll_func, theta0, options);
    
    % Extract results
    hat_r_tars = theta_opt(1);
    hat_v_tars = theta_opt(2);
    
end

function nll = compute_nll(theta, y_time, p_time, A, noise_power, T_fast, T_slow, fc, c, N_fast, N_slow)
    % Compute negative log-likelihood
    % Signal model: y[n,m] = A * p[n-n0] * exp(j*(2*pi*fd*m*T_slow - phi0))
    
    r = theta(1);
    v = theta(2);
    
    
    % Generate signal model
    r_time = generate_signal_model_parametric(r, v, A, p_time, T_fast, T_slow, fc, c, N_fast, N_slow);
    
    % Compute residual
    residual = y_time - r_time;
    
    % Negative log-likelihood: sum of squared errors / (2 * sigma^2)
    nll = sum(abs(residual(:)).^2) / (2 * noise_power);
end

function r_time = generate_signal_model_parametric(r, v, A, p_time, T_fast, T_slow, fc, c, N_fast, N_slow)
    % Generate parametric signal model following:
    % y[n,m] = A * p[n-n0] * exp(j*(2*pi*fd*m*T_slow - phi0))
    %
    % where:
    %   n0 = round(2*R / (c*T_fast)) - delay in chips
    %   fd = 2*v*fc/c - Doppler frequency (Hz)
    %   phi0 = 4*pi*fc*R/c - carrier phase shift (radians)

    % Step 1: Compute delay in chips (round-trip)
    tau = 2 * r / c; % Round-trip delay (seconds)
    n0 = round(tau / T_fast); % Integer delay in chips

    % Step 2: Shift PRBS code by delay
    % p[n - n0] means we shift p to the right by n0 samples
    % In MATLAB: circshift with negative means shift left, positive means shift right
    % We want p[n] -> p[n-n0], which is shifting the sequence to the right
    n0_wrapped = mod(n0, N_fast); % Wrap around for circular shift
    p_shifted = circshift(p_time, n0_wrapped); % Shift right by n0

    % Step 3: Compute Doppler frequency
    fd = -2 * v * fc / c; % Hz

    % Step 4: Compute carrier phase shift
    phi0 = 4 * pi * fc * r / c; % radians

    % Step 5: Generate phase modulation for slow-time
    % Phase varies as: 2*pi*fd*m*T_slow for m = 0, 1, 2, ..., N_slow-1
    m_indices = 0:(N_slow-1); % Slow-time indices
    phase_doppler = 2 * pi * fd * m_indices * T_slow; % 1 x N_slow

    % Step 6: Combine all components
    % y[n,m] = A * p_shifted[n] * exp(j*(phase_doppler[m] - phi0))

    % Create phase matrix: subtract phi0 from each Doppler phase
    phase_total = phase_doppler - phi0; % 1 x N_slow

    % Generate complex exponential
    exp_phase = exp(1j * phase_total); % 1 x N_slow

    % Combine: p_shifted (N_fast x 1) with exp_phase (1 x N_slow)
    % Broadcasting: (N_fast x 1) * (1 x N_slow) -> (N_fast x N_slow)
    r_time = A * p_shifted * exp_phase; % N_fast x N_slow
end