%% ============================================================
%  Reproducing Fig.3 in
%  "MU-MIMO Communications with MIMO Radar:
%   From Co-existence to Joint Transmission"
%  Multi-beam beampatterns: Separated vs Shared deployment
%  (Radar-only vs RadCom)
%  ------------------------------------------------------------
%  依赖: CVX (SDP solver)
% =============================================================

clear; clc; close all;

%% ============= System Parameters (aligned with paper) =======
P0_dBm = 20;                          % total BS power (dBm)
P0     = 10^(P0_dBm/10);              % linear scale
N      = 20;                          % total antennas at BS
NR     = 14;                          % radar antennas (separated)
NC     = N - NR;                      % communication antennas
K      = 4;                           % users
N0_dBm = 0;                           % noise power (dBm)
N0     = 10^(N0_dBm/10);              % linear
lambda = 1;                           % wavelength (normalized)
d      = 0.5*lambda;                  % inter-element spacing (half-λ)

% SINR target (same for all users, 10 dB in the paper)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

% Power split for separated deployment (paper uses roughly half/half)
PR = P0/2;                            % radar power (separated)
PC = P0/2;                            % comm power (separated)

rng(1);  % 固定随机种子，保证复现性（可选）

%% ============= Angle Grid and Steering Vectors =============
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

% Steering vectors for full N-antenna ULA
A_full = zeros(N, M);
for m = 1:M
    A_full(:, m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m)));
end

% For separated case: radar antennas are the first NR, comm are last NC
A1 = A_full(1:NR, :);         % radar array manifold (separated)
A2 = A_full(NR+1:end, :);     % comm array manifold (separated)

%% ============= Desired Multi-beam Beampattern =============
% 5 beams at [-60, -36, 0, 36, 60] degrees (论文设定)
beam_dirs_deg = [-60 -36 0 36 60];

% Ideal beampattern P_tilde(θ): 1 in a small region around each beam, 0 elsewhere
P_tilde = zeros(M, 1);        % 用列向量，方便和 CVX 对齐
main_bw_deg = 6;              % half-width of each mainlobe region (可调)

for b = 1:length(beam_dirs_deg)
    idx = abs(theta_deg - beam_dirs_deg(b)) <= main_bw_deg;
    P_tilde(idx) = 1;
end

%% ============= Random Channel Realization ==================
% H: N x K, Rayleigh flat fading, i.i.d. CN(0,1)
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

% Split into radar part (f_i) and comm part (g_i) for separated deployment
F = H(1:NR, :);     % radar -> user channels (NR x K)
G = H(NR+1:end, :); % comm -> user channels (NC x K)

Bi = cell(K,1);     % Bi = h_i h_i^T for shared case
for i = 1:K
    Bi{i} = H(:,i)*H(:,i).';   % (N x N)
end

%% ============================================================
%  Part 1: Radar-only Beampatterns (Separated and Shared)
% =============================================================

%% ---- 1.1 Separated deployment: Radar-only with ZF constraints (12)
fprintf('Solving radar-only separated (multi-beam) via CVX...\n');
cvx_begin sdp quiet
    variable R1(NR, NR) hermitian semidefinite
    variable alpha_sep
    expression beamp_sep_radar(M)
    
    % Radar beampattern using only radar antennas
    for m = 1:M
        a1m = A1(:, m);
        % 这里直接取实部，beampattern 为实数
        beamp_sep_radar(m) = real( a1m' * R1 * a1m );
    end
    
    minimize( sum_square( alpha_sep * P_tilde - beamp_sep_radar ) )
    subject to
        % per-antenna power for radar part (separated case)
        diag(R1) == (PR/NR) * ones(NR,1);
        alpha_sep >= 0;
        % Zero-forcing: no radar interference to users
        for i = 1:K
            fi = F(:, i);
            trace(fi*fi' * R1) == 0;
        end
cvx_end

%% ---- 1.2 Shared deployment: Radar-only (no null constraints) (9)
fprintf('Solving radar-only shared (multi-beam) via CVX...\n');
cvx_begin sdp quiet
    variable R2(N, N) hermitian semidefinite
    variable alpha_shared
    expression beamp_shared_radar(M)
    
    for m = 1:M
        a_full = A_full(:, m);
        beamp_shared_radar(m) = real( a_full' * R2 * a_full );
    end
    
    minimize( sum_square( alpha_shared * P_tilde - beamp_shared_radar ) )
    subject to
        diag(R2) == (P0/N)*ones(N,1);   % per-antenna power in shared radar-only
        alpha_shared >= 0;
cvx_end

%% ============================================================
%  Part 2: RadCom Joint Transmission
%          Separated deployment (19) and Shared deployment (20) via SDR
% ============================================================

%% === 2.0 预计算一些常数（用于 CVX 中简化表达） ===
% 对于分离式 RadCom:
% S_comm{m} = a2(theta_m) a2(theta_m)^H
S_comm = cell(M,1);
beamp_radar_ref = zeros(M,1);
for m = 1:M
    a2m = A2(:,m);
    S_comm{m} = a2m * a2m';              % (NC x NC), 常数
    a1m = A1(:,m);
    beamp_radar_ref(m) = real( a1m' * R1 * a1m );   % radar-only 参考 beampattern
end

% 用户信道外积（用于 SINR 约束）
GiGiH = cell(K,1);    % NC x NC
FiFiH = cell(K,1);    % NR x NR
HiHiH = cell(K,1);    % N x N  (shared case)
for i = 1:K
    gi = G(:,i);
    fi = F(:,i);
    hi = H(:,i);
    GiGiH{i} = gi * gi';      % (NC x NC), 常数
    FiFiH{i} = fi * fi';      % (NR x NR), 常数
    HiHiH{i} = hi * hi';      % ( N x N ), 常数
end

%% ---- 2.1 Separated deployment: Downlink beamforming (19)
fprintf('Solving RadCom (separated) downlink beamforming via CVX...\n');
cvx_begin sdp quiet
    variable W1(NC,NC) hermitian semidefinite
    variable W2(NC,NC) hermitian semidefinite
    variable W3(NC,NC) hermitian semidefinite
    variable W4(NC,NC) hermitian semidefinite
    variable sig
    
    expression sumW(NC,NC)
    sumW = W1 + W2 + W3 + W4;
    
    % Beampattern for communication part: linear in sumW
    expression beamp_comm_sep(M)
    for m = 1:M
        beamp_comm_sep(m) = real( trace( sumW * S_comm{m} ) );
    end
    
    % Objective: match comm beampattern to scaled radar beampattern
    minimize( sum_square( beamp_comm_sep - sig * beamp_radar_ref ) )
    
    subject to
        % SINR constraints for K=4 users (trace 形式，线性)
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
        
        % Total comm transmit power
        trace(sumW) <= PC;
        sig >= 0;
cvx_end

%% ---- 2.2 Shared deployment: Joint beamforming (20)
fprintf('Solving RadCom (shared) joint beamforming via CVX (SDR)...\n');
cvx_begin sdp quiet
    variable T1(N,N) hermitian semidefinite
    variable T2(N,N) hermitian semidefinite
    variable T3(N,N) hermitian semidefinite
    variable T4(N,N) hermitian semidefinite
    
    expression sumT(N,N)
    sumT = T1 + T2 + T3 + T4;
    
    % Objective: Frobenius norm to match desired radar covariance R2
    minimize( square_pos( norm(sumT - R2, 'fro') ) )
    
    subject to
        % SINR constraints gamma_i >= gamma, i=1..4
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
        
        % Per-antenna power (shared): diag(sumT) = P0/N
        diag(sumT) == (P0/N)*ones(N,1);
cvx_end
%% ============================================================
%  Part 3: Compute and Plot Beampatterns (Radar-only vs RadCom)
% =============================================================

% --- 3.1 Separated deployment: Radar-only vs RadCom ---
% Radar-only covariance (full N×N, comm antennas silent)
C_radar_sep_full = blkdiag(R1, zeros(NC,NC));  % (20x20)

% RadCom covariance (R1 for radar part, sumW for comm part), as (14)
C_radcom_sep = blkdiag(R1, full(W1 + W2 + W3 + W4));

beamp_radar_sep_full = zeros(1,M);
beamp_radcom_sep    = zeros(1,M);
for m = 1:M
    a_full = A_full(:,m);
    beamp_radar_sep_full(m) = real( a_full' * C_radar_sep_full * a_full );
    beamp_radcom_sep(m)     = real( a_full' * C_radcom_sep     * a_full );
end

% --- 3.2 Shared deployment: Radar-only vs RadCom ---
C_radar_shared  = R2;                      % (20x20)
C_radcom_shared = full(T1 + T2 + T3 + T4); % (20x20)

beamp_radar_shared  = zeros(1,M);
beamp_radcom_shared = zeros(1,M);
for m = 1:M
    a_full = A_full(:,m);
    beamp_radar_shared(m)  = real( a_full' * C_radar_shared  * a_full );
    beamp_radcom_shared(m) = real( a_full' * C_radcom_shared * a_full );
end

% === 统一归一化：用四条曲线的“全局最大值”做归一化 ===
all_val = [beamp_radar_sep_full, beamp_radcom_sep, ...
           beamp_radar_shared,  beamp_radcom_shared];
Pmax = max(abs(all_val));    % 全局最大功率

beamp_radar_sep_dB    = 10*log10( abs(beamp_radar_sep_full) / Pmax );
beamp_radcom_sep_dB   = 10*log10( abs(beamp_radcom_sep)    / Pmax );
beamp_radar_shared_dB = 10*log10( abs(beamp_radar_shared)  / Pmax );
beamp_radcom_shared_dB= 10*log10( abs(beamp_radcom_shared) / Pmax );

%% ===================== Plotting (Fig.3 style) =========================
figure;

% --- 左图：Separated Deployment ---
subplot(1,2,1);
plot(theta_deg, beamp_radar_sep_dB,  'LineWidth',1.5); hold on;
plot(theta_deg, beamp_radcom_sep_dB, 'LineWidth',1.5, 'LineStyle','--');
grid on;
xlabel('\theta (deg)');
ylabel('Normalized Beampattern (dB)');
title('Separated Deployment (Multi-beam)');
legend('Radar-Only','RadCom','Location','SouthWest');
ylim([-40 5]);
xlim([-90 90]);

% --- 右图：Shared Deployment ---
subplot(1,2,2);
plot(theta_deg, beamp_radar_shared_dB,  'LineWidth',1.5); hold on;
plot(theta_deg, beamp_radcom_shared_dB, 'LineWidth',1.5, 'LineStyle','--');
grid on;
xlabel('\theta (deg)');
ylabel('Normalized Beampattern (dB)');
title('Shared Deployment (Multi-beam)');
legend('Radar-Only','RadCom','Location','SouthWest');
ylim([-40 5]);
xlim([-90 90]);
