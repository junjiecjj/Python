%% ============================================================
%  Reproduce Fig.3 in
%  "MU-MIMO Communications with MIMO Radar:
%   From Co-existence to Joint Transmission" (Liu et al., TSP'18)
%  Multi-beam beampatterns: Separated vs Shared deployment
%  Radar-Only vs RadCom (SDR, no randomization)
%  ------------------------------------------------------------
%  Needs: CVX (SDP solver)
% =============================================================
clear; clc; close all;

%% ============= System parameters (paper-aligned) =============
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

rng(1);                      % for reproducibility

%% ============= Angle grid and steering matrices =============
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

% Full N-antenna ULA steering
A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m))/lambda);
end

% Separated deployment: first NR are radar, last NC are comm
A1 = A_full(1:NR ,:);   % radar array manifold
A2 = A_full(NR+1:end,:);% comm array manifold

% Precompute S matrices for beampatterns: S(m) = a(theta_m) a(theta_m)^H
S1    = cell(M,1);      % NR x NR, for radar-only (separated)
S2    = cell(M,1);      % NC x NC, for comm (separated)
Sfull = cell(M,1);      % N x N,  for shared deployment

for m = 1:M
    a1m = A1(:,m);
    a2m = A2(:,m);
    afm = A_full(:,m);
    S1{m}    = a1m * a1m';
    S2{m}    = a2m * a2m';
    Sfull{m} = afm * afm';
end

%% ============= Desired multi-beam beampattern ===============
% 5 beams at [-60, -36, 0, 36, 60] degrees
beam_dirs_deg = [-60 -36 0 36 60];
main_bw_deg   = 5;           % half mainlobe width

P_tilde = zeros(M,1);        % desired pattern on angle grid
for b = 1:numel(beam_dirs_deg)
    idx = abs(theta_deg - beam_dirs_deg(b)) <= main_bw_deg;
    P_tilde(idx) = 1;
end

%% ============= Random user channels (Rayleigh) ==============
% H: N x K, BS-to-users channel, CN(0,1)
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

% For separated deployment: split channel into radar/comm parts
F = H(1:NR ,:);   % radar -> users
G = H(NR+1:end,:);% comm  -> users

% Precompute outer-products for SINR constraints
GiGiH = cell(K,1);   % NC x NC
FiFiH = cell(K,1);   % NR x NR
HiHiH = cell(K,1);   % N  x N (shared)
for i = 1:K
    gi = G(:,i);
    fi = F(:,i);
    hi = H(:,i);
    GiGiH{i} = gi * gi';
    FiFiH{i} = fi * fi';
    HiHiH{i} = hi * hi';
end

%% ============================================================
%  Part 1: Radar-Only beampatterns (Separated & Shared)
% =============================================================

%% ---- 1.1 Separated deployment (Eq.12) ----------------------
fprintf('Solving radar-only (separated) via CVX...\n');
cvx_begin sdp quiet
    variable R1(NR,NR) hermitian semidefinite
    variable alpha_sep
    expression beamp_sep_rad(M)
    for m = 1:M
        beamp_sep_rad(m) = real( trace( R1 * S1{m} ) );
    end
    minimize( sum_square( alpha_sep * P_tilde - beamp_sep_rad ) )
    subject to
        % per-antenna power (radar array)
        diag(R1) == (PR/NR)*ones(NR,1);
        alpha_sep >= 0;
        % zero interference from radar to users
        for i = 1:K
            beamp_user = real( trace( R1 * FiFiH{i} ) );
            beamp_user == 0;
        end
cvx_end

% store reference radar-only pattern for separated case
beamp_radar_sep_ref = beamp_sep_rad;

%% ---- 1.2 Shared deployment (Eq.9) --------------------------
fprintf('Solving radar-only (shared) via CVX...\n');
cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable alpha_sh
    expression beamp_sh_rad(M)
    for m = 1:M
        beamp_sh_rad(m) = real( trace( R2 * Sfull{m} ) );
    end
    minimize( sum_square( alpha_sh * P_tilde - beamp_sh_rad ) )
    subject to
        diag(R2) == (P0/N)*ones(N,1);   % per-antenna power
        alpha_sh >= 0;
cvx_end

%% ============================================================
%  Part 2: RadCom joint beamforming (Separated & Shared, SDR)
% =============================================================

%% ---- 2.1 Separated deployment RadCom (Eq.19) ---------------
fprintf('Solving RadCom (separated) via CVX (SDR)...\n');

% radar-only reference beampattern already in beamp_radar_sep_ref

cvx_begin sdp quiet
    variable W1(NC,NC) hermitian semidefinite
    variable W2(NC,NC) hermitian semidefinite
    variable W3(NC,NC) hermitian semidefinite
    variable W4(NC,NC) hermitian semidefinite
    variable sig_sep

    expression sumW(NC,NC)
    sumW = W1 + W2 + W3 + W4;

    % comm beampattern (on comm array) as linear function of sumW
    expression beamp_comm_sep(M)
    for m = 1:M
        beamp_comm_sep(m) = real( trace( sumW * S2{m} ) );
    end

    minimize( sum_square( beamp_comm_sep - sig_sep * beamp_radar_sep_ref ) )
    subject to
        % SINR constraints for K=4 users
        % beta_i = num / den >= gamma
        % => num >= gamma * den
        for i = 1:K
            switch i
                case 1
                    num = real( trace( W1 * GiGiH{i} ) );
                    den = real( trace( W2 * GiGiH{i} ) ...
                              + trace( W3 * GiGiH{i} ) ...
                              + trace( W4 * GiGiH{i} ) ...
                              + trace( R1 * FiFiH{i} ) ) + N0;
                case 2
                    num = real( trace( W2 * GiGiH{i} ) );
                    den = real( trace( W1 * GiGiH{i} ) ...
                              + trace( W3 * GiGiH{i} ) ...
                              + trace( W4 * GiGiH{i} ) ...
                              + trace( R1 * FiFiH{i} ) ) + N0;
                case 3
                    num = real( trace( W3 * GiGiH{i} ) );
                    den = real( trace( W1 * GiGiH{i} ) ...
                              + trace( W2 * GiGiH{i} ) ...
                              + trace( W4 * GiGiH{i} ) ...
                              + trace( R1 * FiFiH{i} ) ) + N0;
                case 4
                    num = real( trace( W4 * GiGiH{i} ) );
                    den = real( trace( W1 * GiGiH{i} ) ...
                              + trace( W2 * GiGiH{i} ) ...
                              + trace( W3 * GiGiH{i} ) ...
                              + trace( R1 * FiFiH{i} ) ) + N0;
            end
            num >= gamma * den;
        end

        % total comm transmit power
        trace(sumW) <= PC;
        sig_sep >= 0;
cvx_end

%% ---- 2.2 Shared deployment RadCom (Eq.20) ------------------
fprintf('Solving RadCom (shared) via CVX (SDR)...\n');
cvx_begin sdp quiet
    variable T1(N,N) hermitian semidefinite
    variable T2(N,N) hermitian semidefinite
    variable T3(N,N) hermitian semidefinite
    variable T4(N,N) hermitian semidefinite

    expression sumT(N,N)
    sumT = T1 + T2 + T3 + T4;

    % objective: make sumT close to radar-only R2 (Frobenius)
    minimize( square_pos( norm(sumT - R2, 'fro') ) )
    subject to
        % SINR constraints (shared) using H
        for i = 1:K
            switch i
                case 1
                    num = real( trace( T1 * HiHiH{i} ) );
                    den = real( trace( T2 * HiHiH{i} ) ...
                              + trace( T3 * HiHiH{i} ) ...
                              + trace( T4 * HiHiH{i} ) ) + N0;
                case 2
                    num = real( trace( T2 * HiHiH{i} ) );
                    den = real( trace( T1 * HiHiH{i} ) ...
                              + trace( T3 * HiHiH{i} ) ...
                              + trace( T4 * HiHiH{i} ) ) + N0;
                case 3
                    num = real( trace( T3 * HiHiH{i} ) );
                    den = real( trace( T1 * HiHiH{i} ) ...
                              + trace( T2 * HiHiH{i} ) ...
                              + trace( T4 * HiHiH{i} ) ) + N0;
                case 4
                    num = real( trace( T4 * HiHiH{i} ) );
                    den = real( trace( T1 * HiHiH{i} ) ...
                              + trace( T2 * HiHiH{i} ) ...
                              + trace( T3 * HiHiH{i} ) ) + N0;
            end
            num >= gamma * den;
        end

        % per-antenna power constraint
        diag(sumT) == (P0/N)*ones(N,1);
cvx_end

%% ============================================================
%  Part 3: Overall beampatterns and Fig.3-style plots
% =============================================================

% --- Separated deployment: overall covariance ---
C_radar_sep = blkdiag(R1, zeros(NC,NC));              % radar-only
C_radcom_sep = blkdiag(R1, full(W1+W2+W3+W4));        % RadCom

beamp_radar_sep   = zeros(1,M);
beamp_radcom_sep  = zeros(1,M);
for m = 1:M
    a_full = A_full(:,m);
    beamp_radar_sep(m)  = real( a_full' * C_radar_sep  * a_full );
    beamp_radcom_sep(m) = real( a_full' * C_radcom_sep * a_full );
end

% --- Shared deployment: overall covariance ---
C_radar_shared  = R2;
C_radcom_shared = full(T1+T2+T3+T4);

beamp_radar_sh   = zeros(1,M);
beamp_radcom_sh  = zeros(1,M);
for m = 1:M
    a_full = A_full(:,m);
    beamp_radar_sh(m)  = real( a_full' * C_radar_shared  * a_full );
    beamp_radcom_sh(m) = real( a_full' * C_radcom_shared * a_full );
end

% === Global normalization (all four curves share one 0 dB reference) ===
all_vals = [beamp_radar_sep, beamp_radcom_sep, ...
            beamp_radar_sh,  beamp_radcom_sh];
Pmax = max(abs(all_vals));

rad_sep_dB   = 10*log10( abs(beamp_radar_sep)  / Pmax );
rc_sep_dB    = 10*log10( abs(beamp_radcom_sep) / Pmax );
rad_sh_dB    = 10*log10( abs(beamp_radar_sh)   / Pmax );
rc_sh_dB     = 10*log10( abs(beamp_radcom_sh)  / Pmax );

%% -------------------- Plot (Fig.3 style) --------------------
figure;

% Left: Separated deployment
subplot(1,2,1);
plot(theta_deg, rad_sep_dB, 'b','LineWidth',1.5); hold on;
plot(theta_deg, rc_sep_dB,  'r--','LineWidth',1.5);
grid on;
xlabel('\theta (deg)');
ylabel('Normalized Beampattern (dB)');
title('Separated Deployment (Multi-beam)');
legend('Radar-Only','RadCom','Location','SouthWest');
xlim([-90 90]); ylim([-40 5]);

% Right: Shared deployment
subplot(1,2,2);
plot(theta_deg, rad_sh_dB, 'b','LineWidth',1.5); hold on;
plot(theta_deg, rc_sh_dB,  'r--','LineWidth',1.5);
grid on;
xlabel('\theta (deg)');
ylabel('Normalized Beampattern (dB)');
title('Shared Deployment (Multi-beam)');
legend('Radar-Only','RadCom','Location','SouthWest');
xlim([-90 90]); ylim([-40 5]);
