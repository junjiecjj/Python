function [RX_opt, Rnk_opt, eta, dbg] = solve_TX_Dinkelbach( ...
    Vcc, R_eta, a_t, b_t, sigma2_tar, u, h_comm, gamma, sigma2_comm, Pt, eta_init, opts)
% Solve the transmit covariance optimization for fixed u by Dinkelbach
% iterations and SDP relaxation.
%
% Variables:
%   RX(:,:,n)      : Total transmit covariance at subcarrier n
%   Rnk(:,:,n,k)   : User-specific covariance at subcarrier n for user k
%
% Inputs:
%   Vcc, R_eta     : Clutter kernel and interference-plus-noise covariance
%   a_t, b_t       : Target steering vectors
%   sigma2_tar     : Target power gain over subcarriers
%   u              : Receive beamformers
%   h_comm         : Communication channel tensor
%   gamma          : SINR requirements
%   sigma2_comm    : Communication noise power
%   Pt             : Total transmit power
%   eta_init       : Initial Dinkelbach parameter
%   opts           : Optimization options
%
% Outputs:
%   RX_opt         : Optimized total transmit covariance
%   Rnk_opt        : Optimized user-specific covariance
%   eta            : Final Dinkelbach parameter
%   dbg            : Debug structure

N = numel(Vcc);
[Nt, ~, K] = size(h_comm);

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
R_eta_sc = s * R_eta;

% Scale the communication part for numerical stability
s_comm = 1e10;
h_comm = sqrt(s_comm) * h_comm;
sigma2_comm = s_comm * sigma2_comm;

dbg = struct();
eta = eta_init;

for it = 1:opts.maxDinkel
    cvx_solver mosek
    cvx_begin sdp quiet
    cvx_precision high
    variable RX(Nt, Nt, N) hermitian semidefinite
    variable Rnk(Nt, Nt, N, K) hermitian semidefinite

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

    % Per-user SINR constraints
    for n = 1:N
        for k = 1:K
            hk = h_comm(:, n, k);
            Hk = hk * hk';
            (1 + 1 / gamma(n, k)) * real(trace(Hk * Rnk(:, :, n, k))) ...
                >= real(trace(Hk * RX(:, :, n))) + sigma2_comm;
        end
    end

    % Decompose RX into communication and sensing parts
    for n = 1:N
        expression sumR
        sumR = 0;
        for k = 1:K
            sumR = sumR + Rnk(:, :, n, k);
        end
        RX(:, :, n) - sumR == semidefinite(Nt);
    end
    cvx_end

    RX_opt = RX;
    Rnk_opt = Rnk;

    % Dinkelbach update
    num = 0;
    den = 0;
    for n = 1:N
        num = num + real(trace(At{n} * RX_opt(:, :, n)));
        den = den + real(trace(Ac{n} * RX_opt(:, :, n))) ...
            + real(u(:, n)' * R_eta_sc(:, :, n) * u(:, n));
    end
    eta_new = num / den;

    if opts.verbose
        eta_dB = 10 * log10(eta);
        eta_new_dB = 10 * log10(eta_new);
        fprintf('  [Dinkel %2d] obj=%.3e, eta=%.4e (%.2f dB) -> %.4e (%.2f dB)\n', ...
            it, cvx_optval, eta, eta_dB, eta_new, eta_new_dB);
    end

    rel_eta = abs(eta_new - eta) / max(abs(eta), 1e-30);
    if rel_eta <= opts.tolDinkel_eta
        eta = eta_new;
        break;
    end
    eta = eta_new;
end

dbg.At = At;
dbg.Ac = Ac;
end