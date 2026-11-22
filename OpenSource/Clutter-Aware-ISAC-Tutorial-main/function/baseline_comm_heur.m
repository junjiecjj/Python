function out = baseline_comm_heur(para, H_comm, Vcc, R_eta, tarPara, a_t, b_t)
% Compute two baseline designs:
% 1) communication-only transceiver design;
% 2) heuristic ISAC transceiver design.
%
% Inputs:
%   para     : System parameter structure
%   H_comm   : Communication channel tensor
%   Vcc      : Clutter kernel over subcarriers
%   R_eta    : Interference-plus-noise covariance over subcarriers
%   tarPara  : Target parameter structure
%   a_t      : Target transmit steering vectors
%   b_t      : Target receive steering vectors
%
% Output:
%   out      : Structure containing baseline covariances, beamformers,
%              and SCNR values

Nt = para.Nt;
Nr = para.Nr;
N  = para.N;
Ku = para.Ku;
Ns = para.Ns;
Pt = para.Pt;
sigma2_tar = abs(tarPara.alpha_mn).^2;
chi_n = @(n) 1 + n * para.deltaf / para.fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);

% 1) Communication-only transmit design
W_comm = zeros(Nt, Ns, N);
RX_comm = zeros(Nt, Nt, N);
for n = 1:N
    Hn = squeeze(H_comm(:, n, :));

    % Communication streams: MRT along each user channel
    for k = 1:Ku
        hk = Hn(:, k);
        hk_norm = norm(hk);
        if hk_norm > 0
            W_comm(:, k, n) = hk / hk_norm;
        else
            W_comm(:, k, n) = zeros(Nt, 1);
        end
    end

    % No sensing streams for the communication-only baseline
    if Ns > Ku
        W_comm(:, Ku+1:Ns, n) = 0;
    end
    power_now = real(trace(W_comm(:, :, n) * W_comm(:, :, n)'));
    if power_now > 0
        W_comm(:, :, n) = W_comm(:, :, n) * sqrt((Pt / N) / power_now);
    end
    RX_comm(:, :, n) = W_comm(:, :, n) * W_comm(:, :, n)';
end

% MVDR receive beamformer for the communication-only baseline
u_comm = zeros(Nr, N);
for n = 1:N
    Rcc_n = kernel_apply(Vcc{n}, RX_comm(:, :, n), Nr);
    RI_n = Rcc_n + R_eta(:, :, n);
    u_comm(:, n) = mvdr_beamformer(RI_n, b_t{n}, 1e-10);
end
SCNR_comm = compute_SCNR(RX_comm, u_comm, Vcc, R_eta, sigma2_tar, a_t, b_t);

% 2) Heuristic ISAC transmit design
W_heur = zeros(Nt, Ns, N);
RX_heur = zeros(Nt, Nt, N);
for n = 1:N
    Hn = squeeze(H_comm(:, n, :));

    % Communication streams: MRT along each user channel
    for k = 1:Ku
        hk = Hn(:, k);
        hk_norm = norm(hk);
        if hk_norm > 0
            W_heur(:, k, n) = hk / hk_norm;
        else
            W_heur(:, k, n) = zeros(Nt, 1);
        end
    end

    % Sensing streams: target steering vectors
    for s = 1:(Ns-Ku)
        idx = mod(s-1, tarPara.M) + 1;
        W_heur(:, Ku+s, n) = a_n(tarPara.theta(idx), n-1);
    end
    power_now = real(trace(W_heur(:, :, n) * W_heur(:, :, n)'));
    if power_now > 0
        W_heur(:, :, n) = W_heur(:, :, n) * sqrt((Pt / N) / power_now);
    end

    RX_heur(:, :, n) = W_heur(:, :, n) * W_heur(:, :, n)';
end

% MVDR receive beamformer for the heuristic baseline
u_heur = zeros(Nr, N);
for n = 1:N
    Rcc_n = kernel_apply(Vcc{n}, RX_heur(:, :, n), Nr);
    RI_n = Rcc_n + R_eta(:, :, n);
    u_heur(:, n) = mvdr_beamformer(RI_n, b_t{n}, 1e-10);
end
SCNR_heur = compute_SCNR(RX_heur, u_heur, Vcc, R_eta, sigma2_tar, a_t, b_t);

% Outputs
out.RX_comm = RX_comm;
out.u_comm = u_comm;
out.SCNR_comm = SCNR_comm;

out.RX_heur = RX_heur;
out.u_heur = u_heur;
out.SCNR_heur = SCNR_heur;
end