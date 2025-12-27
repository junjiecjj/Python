% =============================================================
% Fig.4(a) - Separated Deployment (3 dB Beampattern)
% MU-MIMO Communications With MIMO Radar: From Co-Existence to Joint Transmission
%
% - Radar-only: solve Problem (13) to get R1 (NRxNR)
% - RadCom:     solve SDR of Problem (19) to get Wi (NCxNC), then
%               do rank-1 extraction: Wi ≈ wi*wi^H (dominant eigenpair)
% - Plot:       Array Gain (dBi) using per-antenna average power normalization
%
% NOTE:
%   Rank-1 extraction may slightly violate SINR constraints; script prints SINR check.
% =============================================================
clear; clc; close all;

%% ============= System parameters (same as Fig.3/Fig.4) =============
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear
N      = 20;                 % total antennas at BS
NR     = 14;                 % radar antennas (separated)
NC     = N - NR;             % comm antennas (separated)
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);     % linear

lambda = 1;                  % normalized wavelength
d      = 0.5*lambda;         % half-wavelength spacing

% SINR target for all users (10 dB)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

% Power split for separated deployment: half radar, half comm
PR = P0/2;                   % radar power (separated)
PC = P0/2;                   % comm power (separated)

rng(1);                      % reproducibility

%% ============= Angle grid and steering matrices =============
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

% Full N-antenna ULA steering (A_full: N x M)
A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*(d/lambda)*(0:N-1)' * sin(theta_rad(m)));
end

% Separated deployment: first NR are radar, last NC are comm
A1 = A_full(1:NR ,:);        % radar subarray manifold (NR x M)
A2 = A_full(NR+1:end,:);     % comm  subarray manifold (NC x M)

%% ============= Random user channels (Rayleigh) ==============
% H: N x K, BS-to-users channel, CN(0,1)
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

% For separated deployment: split channel into radar/comm parts
F = H(1:NR ,:);              % radar -> users (NR x K)
G = H(NR+1:end,:);           % comm  -> users (NC x K)

% Precompute outer-products for SINR constraints
GiGiH = cell(K,1);           % NC x NC, gi gi^H
FiFiH = cell(K,1);           % NR x NR, fi fi^H
for i = 1:K
    gi        = G(:,i);
    fi        = F(:,i);
    GiGiH{i}  = gi * gi';
    FiFiH{i}  = fi * fi';
end

% =============================================================
%  Part 1: Radar-Only (Separated) via Problem (13) - 3 dB beam
% =============================================================

% ---- 1.1 Define main-beam and sidelobe regions ----
theta0   = 0;                 % main-beam center (deg)
bw_3dB   = 10;                % 3 dB beamwidth (deg)
theta1   = theta0 - bw_3dB/2; % -5 deg
theta2   = theta0 + bw_3dB/2; % +5 deg

% Indices on the angle grid
[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

% Sidelobe region Ω: all angles outside [theta1, theta2]
idx_sidelobe = find( (theta_deg < theta1) | (theta_deg > theta2) );

% Steering vectors for the three key angles (radar subarray)
a1_0 = A1(:,idx0);
a1_1 = A1(:,idx1);
a1_2 = A1(:,idx2);

fprintf('Solving radar-only 3dB (Separated, Problem 13) via CVX...\n');
cvx_begin sdp quiet
    variable R1(NR,NR) hermitian semidefinite
    variable t_sep

    minimize( -t_sep )
    subject to
        % Per-antenna power on radar subarray
        diag(R1) == (PR/NR) * ones(NR,1);

        % 3 dB constraints (half-power at theta1/theta2)
        main0 = real( a1_0' * R1 * a1_0 );
        left  = real( a1_1' * R1 * a1_1 );
        right = real( a1_2' * R1 * a1_2 );
        left  == 0.5 * main0;
        right == 0.5 * main0;

        % Sidelobe suppression: main0 - sidelobe >= t_sep for θ in Ω
        for mm = idx_sidelobe
            a1m   = A1(:,mm);
            sidel = real( a1m' * R1 * a1m );
            main0 - sidel >= t_sep;
        end

        % Zero interference from radar to users: tr(R1 * fi fi^H) = 0
        for i = 1:K
            real( trace( R1 * FiFiH{i} ) ) == 0;
        end
cvx_end

% Radar-only beampattern (full N aperture, comm part = 0)
C_radar_sep     = blkdiag(R1, zeros(NC,NC));  % N x N
beamp_radar_sep = zeros(1,M);
for m = 1:M
    af = A_full(:,m);
    beamp_radar_sep(m) = real( af' * C_radar_sep * af );
end

% =============================================================
%  Part 2: RadCom (Separated) via SDR of Problem (19) + rank-1 extraction
% =============================================================
fprintf('Solving RadCom (Separated, Problem 19 - SDR) via CVX...\n');

cvx_begin sdp quiet
    variable W(NC,NC,K) hermitian semidefinite
    variable sig_sep

    expression sumW(NC,NC)
    sumW = 0;
    for k = 1:K
        sumW = sumW + W(:,:,k);
    end

    % diag(A2^H sumW A2) - sig * diag(A1^H R1 A1)
    C_comm     = A2' * sumW * A2;      % M x M
    C_radar_A1 = A1' * R1   * A1;      % M x M
    diff_vec   = real( diag( C_comm - sig_sep * C_radar_A1 ) );

    minimize( sum_square(diff_vec) )
    subject to
        % SINR constraints: beta_i >= gamma
        % beta_i = gi^H Wi gi / (gi^H (sum_{k!=i} Wk) gi + fi^H R1 fi + N0)
        for i = 1:K
            num = real( trace( W(:,:,i) * GiGiH{i} ) );

            den = 0;
            for k = 1:K
                if k ~= i
                    den = den + real( trace( W(:,:,k) * GiGiH{i} ) );
                end
            end
            den = den + real( trace( R1 * FiFiH{i} ) ) + N0;

            num >= gamma * den;
        end

        % Comm power budget
        real(trace(sumW)) <= PC;

        % sigma >= 0
        sig_sep >= 0;
cvx_end

% ---------- Rank-1 extraction: Wi ≈ wi*wi^H (dominant eigenpair) ----------
W_rank1 = zeros(NC,NC,K);
w_vecs  = zeros(NC,K);

for i = 1:K
    Wi = full(W(:,:,i));
    Wi = (Wi + Wi')/2;  % enforce Hermitian numerically

    [V,D] = eig(Wi,'vector');
    [lam_max, idx] = max(real(D));
    v_max = V(:,idx);

    lam_max = max(lam_max, 0);          % avoid tiny negative due to numerics
    wi = sqrt(lam_max) * v_max;         % wi such that wi*wi^H has eigenvalue lam_max
    w_vecs(:,i) = wi;

    W_rank1(:,:,i) = wi * wi';
end

sumW_rank1 = zeros(NC,NC);
for i = 1:K
    sumW_rank1 = sumW_rank1 + W_rank1(:,:,i);
end

% Power fix (optional but recommended): enforce trace(sumWi) <= PC
pow_comm_rank1 = real(trace(sumW_rank1));
if pow_comm_rank1 > PC*(1+1e-9)
    scale = sqrt(PC / pow_comm_rank1);   % scale beamformers
    for i = 1:K
        w_vecs(:,i)      = scale * w_vecs(:,i);
        W_rank1(:,:,i)   = w_vecs(:,i) * w_vecs(:,i)';
    end
    sumW_rank1 = zeros(NC,NC);
    for i = 1:K
        sumW_rank1 = sumW_rank1 + W_rank1(:,:,i);
    end
    fprintf('[Rank-1] Comm power scaled to meet PC. scale=%.4f\n', scale);
end

% SINR check after rank-1 extraction (may violate slightly)
beta_rank1 = zeros(K,1);
for i = 1:K
    num = real(trace(W_rank1(:,:,i) * GiGiH{i}));
    den = 0;
    for k = 1:K
        if k ~= i
            den = den + real(trace(W_rank1(:,:,k) * GiGiH{i}));
        end
    end
    den = den + real(trace(R1 * FiFiH{i})) + N0;
    beta_rank1(i) = num / max(den,1e-12);
end
fprintf('[Rank-1] SINR after extraction (dB): ');
fprintf('%.2f  ', 10*log10(beta_rank1));
fprintf('\n');
if any(beta_rank1 < gamma*(1-1e-6))
    fprintf('WARNING: Some SINR constraints are not met after rank-1 extraction.\n');
    fprintf('         (Paper uses rank-1 approximation/randomization; you can add Gaussian randomization if needed.)\n');
end

% Overall covariance for RadCom (Separated): blkdiag(R1, sumWi_rank1)
C_radcom_sep = blkdiag(R1, sumW_rank1);

beamp_radcom_sep = zeros(1,M);
for m = 1:M
    af = A_full(:,m);
    beamp_radcom_sep(m) = real( af' * C_radcom_sep * af );
end

% =============================================================
%  Part 3: Array Gain normalization -> dBi, PSLR, Plot (Fig.4(a))
% =============================================================

% Array Gain (linear): G(theta) = P(theta) / (avg per-antenna power)
% Radar-only: uses NR antennas with power PR => avg PR/NR
G_radar_lin  = beamp_radar_sep(:)  / (PR/NR);

% RadCom: total uses N antennas with total power PR+PC = P0 => avg P0/N
G_radcom_lin = beamp_radcom_sep(:) / (P0/N);

% Convert to dBi
G_radar_dBi  = 10*log10(G_radar_lin  + 1e-12);
G_radcom_dBi = 10*log10(G_radcom_lin + 1e-12);

% PSLR: mainlobe region [theta1, theta2]
idx_main = (theta_deg >= theta1) & (theta_deg <= theta2);
idx_side = ~idx_main;

PSLR_radar_dB = max(G_radar_dBi(idx_main))  - max(G_radar_dBi(idx_side));
PSLR_radc_dB  = max(G_radcom_dBi(idx_main)) - max(G_radcom_dBi(idx_side));
fprintf('PSLR (Radar-Only, Separated 3dB) : %.2f dB\n', PSLR_radar_dB);
fprintf('PSLR (RadCom, Separated 3dB)     : %.2f dB\n', PSLR_radc_dB);

% Plot
figure;
plot(theta_deg, G_radar_dBi,  'b--','LineWidth',1.8); hold on;
plot(theta_deg, G_radcom_dBi, 'r','LineWidth',1.8);
grid on;
xlabel('Angle (Degree)');
ylabel('Array Gain (dBi)');
title('Separated Deployment - 3 dB Beampattern (Fig.4(a))');
legend('Radar-Only','RadCom (rank-1 approx)','Location','Best');
xlim([-90 90]);
ylim([-20 10]); % 可按论文视觉效果微调/或注释掉让其自动
