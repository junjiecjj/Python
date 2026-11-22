function [RX_opt, eta, dbg] = solve_TX_Dinkelbach_radar( ...
    Vcc, R_eta, a_t, b_t, sigma2_tar, u, Pt, eta_init, opts)
% Solve the radar-only transmit covariance optimization for fixed u by
% Dinkelbach iterations and SDP relaxation.
%
% Variable:
%   RX(:,:,n) : Total transmit covariance at subcarrier n
%
% Inputs:
%   Vcc, R_eta   : Clutter kernel and interference-plus-noise covariance
%   a_t, b_t     : Target steering vectors
%   sigma2_tar   : Target power gain over subcarriers
%   u            : Receive beamformers
%   Pt           : Total transmit power
%   eta_init     : Initial Dinkelbach parameter
%   opts         : Optimization options
%
% Outputs:
%   RX_opt       : Optimized transmit covariance
%   eta          : Final Dinkelbach parameter
%   dbg          : Debug structure

N = numel(Vcc);
Nt = size(a_t{1}, 1);

% Precompute target-reward and clutter-penalty matrices
At = cell(N, 1);
Ac = cell(N, 1);
for n = 1:N
    at = a_t{n};
    bt = b_t{n};
    hn = abs(u(:, n)' * bt)^2;
    At{n} = sigma2_tar(n) * hn * (at * at');
    Ac{n} = build_Ac(Vcc{n}, u(:, n), Nt);
end

% Scale the objective for numerical stability
den0 = 0;
for n = 1:N
    den0 = den0 + real(u(:, n)' * R_eta(:, :, n) * u(:, n));
end
den0 = den0 / N;
if den0 <= 0
    s = 1;
else
    s = 1e-8 / den0;
end
for n = 1:N
    At{n} = s * At{n};
    Ac{n} = s * Ac{n};
end
R_eta = s * R_eta;

eta = eta_init;
dbg = struct();
dbg.At = At;
dbg.Ac = Ac;

for it = 1:opts.maxDinkel
    cvx_solver mosek
    cvx_begin sdp quiet
    cvx_precision high
    variable RX(Nt, Nt, N) hermitian semidefinite

    expression obj
    obj = 0;
    for n = 1:N
        obj = obj + real(trace((At{n} - eta * Ac{n}) * RX(:, :, n)));
    end
    maximize(obj)

    % Total transmit power constraint
    expression pow
    pow = 0;
    for n = 1:N
        pow = pow + trace(RX(:, :, n));
    end
    pow <= Pt;
    cvx_end

    RX_opt = RX;

    % Dinkelbach update
    num = 0;
    den = 0;
    for n = 1:N
        num = num + real(trace(At{n} * RX_opt(:, :, n)));
        den = den + real(trace(Ac{n} * RX_opt(:, :, n))) ...
            + real(u(:, n)' * R_eta(:, :, n) * u(:, n));
    end
    eta_new = num / den;

    if opts.verbose
        fprintf('  [Dinkel %2d] obj=%.3e, eta=%.4e (%.2f dB) -> %.4e (%.2f dB)\n', ...
            it, cvx_optval, eta, 10 * log10(eta), eta_new, 10 * log10(eta_new));
    end

    relchg = abs(eta_new - eta) / abs(eta);
    eta = eta_new;
    if relchg < opts.tolDinkel_eta
        break;
    end
end
end