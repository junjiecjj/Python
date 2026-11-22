function out = solve_Radar_AO(para, sigma2_tar, Vcc, R_eta, a_t, b_t, opts)
% Solve the radar-only design by alternating optimization (AO).
%
% Inputs:
%   para       : System parameter structure
%   sigma2_tar : Target power gain over subcarriers
%   Vcc        : Clutter kernel over subcarriers
%   R_eta      : Interference-plus-noise covariance over subcarriers
%   a_t, b_t   : Target steering vectors
%   opts       : Optimization options
%
% Output:
%   out        : Structure containing optimized covariances, beamformers,
%                histories, and solver diagnostics

Nt = para.Nt;
N = para.N;
Ns = para.Ns;
Nr = para.Nr;
Pt = para.Pt;

% Default options
if ~isfield(opts, 'maxAO');       opts.maxAO = 20; end
if ~isfield(opts, 'maxDinkel');   opts.maxDinkel = 30; end
if ~isfield(opts, 'tolAO_dB');    opts.tolAO_dB = 0.1; end
if ~isfield(opts, 'tolDinkel');   opts.tolDinkel = 1e-4; end
if ~isfield(opts, 'lambdaRidge'); opts.lambdaRidge = 1e-10; end
if ~isfield(opts, 'verbose');     opts.verbose = true; end

% Initialize transmit covariance from random sensing beams
W_sens = (randn(Nt, Ns, N) + 1j * randn(Nt, Ns, N)) / sqrt(2);
W_sens = W_sens * sqrt(Pt / norm(W_sens, 'fro')^2);
RX = zeros(Nt, Nt, N);
for n = 1:N
    RX(:, :, n) = W_sens(:, :, n) * W_sens(:, :, n)';
end

% Initialize receive beamformers by MVDR
u = zeros(Nr, N);
for n = 1:N
    Rcc_n = kernel_apply(Vcc{n}, RX(:, :, n), Nr);
    RI_n = Rcc_n + R_eta(:, :, n);
    u(:, n) = mvdr_beamformer(RI_n, b_t{n}, opts.lambdaRidge);
end

scnr0 = compute_SCNR(RX, u, Vcc, R_eta, sigma2_tar, a_t, b_t);
eta_init = scnr0;
eta_hist = zeros(opts.maxAO + 1, 1);
scnr_hist = zeros(opts.maxAO + 1, 1);
eta_hist_dB = zeros(opts.maxAO + 1, 1);
scnr_hist_dB = zeros(opts.maxAO + 1, 1);
eta_hist(1) = eta_init;
scnr_hist(1) = scnr0;
eta_hist_dB(1) = 10 * log10(eta_init);
scnr_hist_dB(1) = 10 * log10(scnr0);

if opts.verbose
    fprintf('[AO %2d] eta=%.4e (%.2f dB), SCNR=%.4e (%.2f dB)\n', ...
        0, eta_init, eta_hist_dB(1), scnr0, scnr_hist_dB(1));
end

idx = 1;
for itAO = 1:opts.maxAO
    % 1) Update the receive beamformer by MVDR
    for n = 1:N
        Rcc_n = kernel_apply(Vcc{n}, RX(:, :, n), Nr);
        RI_n = Rcc_n + R_eta(:, :, n);
        u(:, n) = mvdr_beamformer(RI_n, b_t{n}, opts.lambdaRidge);
    end

    % 2) Update the transmit covariance by radar-only Dinkelbach-SDP
    [RX_new, eta, dbg] = solve_TX_Dinkelbach_radar( ...
        Vcc, R_eta, a_t, b_t, sigma2_tar, u, Pt, eta_init, opts);

    % 3) Evaluate the updated SCNR
    scnr = compute_SCNR(RX_new, u, Vcc, R_eta, sigma2_tar, a_t, b_t);

    idx = itAO + 1;
    eta_hist(idx) = eta;
    scnr_hist(idx) = scnr;
    eta_hist_dB(idx) = 10 * log10(eta);
    scnr_hist_dB(idx) = 10 * log10(scnr);
    if opts.verbose
        fprintf('[AO %2d] eta=%.4e (%.2f dB), SCNR=%.4e (%.2f dB)\n', ...
            itAO, eta, eta_hist_dB(idx), scnr, scnr_hist_dB(idx));
    end
    RX = RX_new;
    eta_init = eta;

    % Stop when the SCNR change in dB becomes sufficiently small
    d_scnr_dB = abs(scnr_hist_dB(idx) - scnr_hist_dB(idx-1));
    if d_scnr_dB < opts.tolAO_dB
        break;
    end
end

% Truncate histories
T = idx;
eta_hist = eta_hist(1:T);
scnr_hist = scnr_hist(1:T);
eta_hist_dB = eta_hist_dB(1:T);
scnr_hist_dB = scnr_hist_dB(1:T);

out.RX = RX;
out.u = u;
out.eta_hist = eta_hist;
out.scnr_hist = scnr_hist;
out.eta_hist_dB = eta_hist_dB;
out.scnr_hist_dB = scnr_hist_dB;
out.dbg_last = dbg;
end