% =============================================================
%  Fig.4(a) - Separated Deployment (3 dB Beampattern, Array Gain)
% =============================================================
clear; clc; close all;

%% ============= System parameters (same as Fig.3) =============
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

% Full N-antenna ULA steering (A_full: N x M)
A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m))/lambda);
end

% Separated deployment: first NR are radar, last NC are comm
A1 = A_full(1:NR ,:);   % radar array manifold
A2 = A_full(NR+1:end,:);% comm  array manifold

%% ============= Random user channels (Rayleigh) ==============
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);  % N x K

F = H(1:NR ,:);   % radar -> users
G = H(NR+1:end,:);% comm  -> users

GiGiH = cell(K,1);
FiFiH = cell(K,1);
HiHiH = cell(K,1);
for i = 1:K
    gi = G(:,i);
    fi = F(:,i);
    hi = H(:,i);
    GiGiH{i} = gi * gi';
    FiFiH{i} = fi * fi';
    HiHiH{i} = hi * hi';
end

% =============================================================
%  Part 1: Radar-Only (Separated) via Problem (13) - 3 dB beam
% =============================================================

% ---- 1.1 Define main-beam and sidelobe regions ----
theta0   = 0;     % main-beam center (deg)
bw_3dB   = 10;    % 3 dB beamwidth (deg)
theta1   = theta0 - bw_3dB/2;   % left  3 dB point (-5)
theta2   = theta0 + bw_3dB/2;   % right 3 dB point (+5)

% Indices on the angle grid (θ0, θ1, θ2)
[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

% Sidelobe region 𝕌 = 所有在 [θ1, θ2] 之外的角度
idx_sidelobe = find( (theta_deg < theta1) | (theta_deg > theta2) );

% Steering vectors for the three key angles (only radar sub-array)
a1_0 = A1(:,idx0);   % θ0
a1_1 = A1(:,idx1);   % θ1
a1_2 = A1(:,idx2);   % θ2

fprintf('Solving radar-only 3dB (separated, Problem 13) via CVX...\n');
cvx_begin sdp quiet
    variable R1(NR,NR) hermitian semidefinite
    variable t_sep

    % Objective: maximize t  <=> minimize -t
    minimize( -t_sep )

    subject to
        % (i) Per-antenna power on radar array: diag(R1) = PR/NR
        diag(R1) == (PR/NR) * ones(NR,1);

        % (ii) 3dB constraints of Stoica-formulation (式(10)):
        %      P(θ1) = P(θ2) = 0.5 * P(θ0)，这里的 P(θ) = a^H R1 a
        main0 = real( a1_0' * R1 * a1_0 );
        left  = real( a1_1' * R1 * a1_1 );
        right = real( a1_2' * R1 * a1_2 );
        left  == 0.5 * main0;
        right == 0.5 * main0;

        % (iii) Sidelobe suppression:
        %       P(θ0) - P(θm) >= t_sep, ∀θm ∈ 𝕌
        for mm = idx_sidelobe
            a1m   = A1(:,mm);
            sidel = real( a1m' * R1 * a1m );
            main0 - sidel >= t_sep;
        end

        % (iv) Zero interference from radar to users:
        %      tr(f_i^* f_i^T R1) = 0, ∀i
        for i = 1:K
            real( trace( R1 * FiFiH{i} ) ) == 0;
        end
cvx_end

% ---- 1.2 计算雷达-only 的全阵列 beampattern（用于画 Fig.4(a)）----
C_radar_sep = blkdiag(R1, zeros(NC,NC));  % overall covariance (N x N)
beamp_radar_sep = zeros(1,M);
for m = 1:M
    af = A_full(:,m);
    beamp_radar_sep(m) = real( af' * C_radar_sep * af );
end

% =============================================================
%  Part 2: RadCom (Separated, Problem 19) matching radar 3 dB
%        
% =============================================================

fprintf('Solving RadCom (separated, Problem 19) via CVX...\n');

cvx_begin sdp quiet
    variable W1(NC,NC) hermitian semidefinite
    variable W2(NC,NC) hermitian semidefinite
    variable W3(NC,NC) hermitian semidefinite
    variable W4(NC,NC) hermitian semidefinite
    variable sig_sep

    expression sumW(NC,NC)
    sumW = W1 + W2 + W3 + W4;

    C_comm     = A2' * sumW * A2;      % M x M
    C_radar_A1 = A1' * R1   * A1;      % M x M

    diff_vec = real( diag( C_comm - sig_sep * C_radar_A1 ) );   % M×1

    minimize( sum_square( diff_vec ) )

    subject to
        % ----- SINR constraints: β_i >= γ -----
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

        % Total comm power
        trace(sumW) <= PC;

        % σ >= 0
        sig_sep >= 0;
cvx_end

C_radcom_sep = blkdiag(R1, full(W1+W2+W3+W4));

beamp_radcom_sep = zeros(1,M);
for m = 1:M
    af = A_full(:,m);
    beamp_radcom_sep(m) = real( af' * C_radcom_sep * af );
end

% =============================================================
%  Part 3: 阵列增益（Array Gain）转为 dBi 并绘制 Fig.4(a)
% =============================================================

% Radar-only：只 NR 根雷达阵元发 PR，总平均每根天线 PR/NR
G_radar_lin  = beamp_radar_sep.'  / (PR/NR);   % M×1


% RadCom：整体 N 根天线发 P0，这里按 P0/N 作为每根平均功率
G_radcom_lin = beamp_radcom_sep.' / (P0/N);    % M×1



%G_radar_dBi_raw  = 10*log10(G_radar_lin  + 1e-12);
%G_radcom_dBi_raw = 10*log10(G_radcom_lin + 1e-12);
G_radar_dBi_raw  = 10*log10(G_radar_lin  + 1e-12);
G_radcom_dBi_raw = 10*log10(G_radcom_lin + 1e-12);


figure;
plot(theta_deg,  G_radar_dBi_raw,  'b--','LineWidth',1.8); hold on;
plot(theta_deg, G_radcom_dBi_raw, 'r','LineWidth',1.8);

grid on; xlabel('Angle (Degree)');
ylabel('Array Gain (dBi)');
title('Separated Deployment - 3 dB Beampattern (Fig.4(a), Array Gain)');
legend('Radar-Only','RadCom','Location','Best');
ylim([-20 10]);
xlim([-90 90]);
