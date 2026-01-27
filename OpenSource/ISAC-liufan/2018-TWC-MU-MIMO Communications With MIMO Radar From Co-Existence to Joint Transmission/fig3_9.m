% =============================================================
%  Fig.3(b) - Shared Deployment (Radar-Only & RadCom)
%  MU-MIMO Communications with MIMO Radar:
%  From Co-existence to Joint Transmission (Liu et al., TSP 2018)
%
%  - Shared deployment: all N antennas used for Radar + Comm
%  - Radar-Only: multi-beam beampattern via Problem (9)
%  - RadCom: shared beamforming via Problem (20) (SDR)
% =============================================================
clear; clc; close all;

%% ============= System parameters =============
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear total power
N      = 20;                 % total antennas at BS
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);     % linear

lambda = 1;                  % normalized wavelength
d      = 0.5*lambda;         % half-wavelength spacing

% SINR target for all users (10 dB)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

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

% Precompute S matrices for beampatterns: S(m) = a(theta_m) a(theta_m)^H
Sfull = cell(M,1);      % N x N
for m = 1:M
    afm       = A_full(:,m);
    Sfull{m}  = afm * afm';
end

%% ============= Desired multi-beam beampattern P_tilde(θ) =============
% 5 beams at [-60, -36, 0, 36, 60] degrees
beam_dirs_deg = [-60 -36 0 36 60];
main_bw_deg   = 3;           % half mainlobe width for each beam (可调)

P_tilde = zeros(M,1);        % desired pattern on angle grid
for b = 1:numel(beam_dirs_deg)
    idx = abs(theta_deg - beam_dirs_deg(b)) <= main_bw_deg;
    P_tilde(idx) = 3;        % 给每个波束一个统一高度（可调）
end

%% ============= Random user channels (Rayleigh) ==============
% H: N x K, BS-to-users channel, CN(0,1)
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

% For SINR constraints: HiHiH{i} = h_i h_i^H
HiHiH = cell(K,1);   % N x N
for i = 1:K
    hi = H(:,i);
    HiHiH{i} = hi * hi';
end

% =============================================================
%  Part 1: Radar-Only Beampattern (Shared, N antennas) via (9)
% =============================================================
% 对应式 (9)：多波束 MIMO 雷达方向图设计
% min_{alpha,R2} sum_m | alpha * P_tilde(theta_m) - a^H R2 a |^2
% s.t. diag(R2) = P0/N * 1, R2 >= 0, Hermitian, alpha >= 0

cvx_begin sdp quiet
    variable R2(N,N) hermitian
    variable alpha_rad_nonneg
    expression err(M,1)

    for m = 1:M
        % a^H R2 a = trace(R2 * a a^H) = trace(R2 * Sfull{m})
        err(m) = alpha_rad_nonneg * P_tilde(m) - trace(R2 * Sfull{m});
    end

    minimize( sum_square_abs(err) )
    subject to
        diag(R2) == (P0/N) * ones(N,1);  % 每天线平均功率 P0/N
        R2 >= 0;                         % PSD: R2 ⪰ 0
        alpha_rad_nonneg >= 0;
cvx_end

alpha_opt = alpha_rad_nonneg;

% 雷达-only方向图：P_radar_only(theta) = a^H R2 a
P_radar_only = zeros(M,1);
for m = 1:M
    P_radar_only(m) = real(trace(R2 * Sfull{m}));
end

% “理想”方向图：alpha_opt * P_tilde(theta)
P_ideal = alpha_opt * P_tilde;

% =============================================================
%  Part 2: RadCom Beampattern (Shared, constrained SDR) via (20)
% =============================================================
% 对应式 (20)：共享天线部署的联合波束优化
% (20a) min || sum_i T_i - R2 ||_F^2
% (20b) γ_i >= gamma,  γ_i 用 (7) 定义
% (20c) diag(sum_i T_i) = P0/N * 1
% (20d) T_i >= 0, Hermitian，rank(T_i)=1 此处用 SDR 放松

cvx_begin sdp quiet
    % 这里 K=4，直接定义 4 个 T_i，Hermitian
    variable T1(N,N) hermitian
    variable T2(N,N) hermitian
    variable T3(N,N) hermitian
    variable T4(N,N) hermitian

    % sum_i T_i
    expression Tsum(N,N)
    Tsum = T1 + T2 + T3 + T4;

    % ===== (20a) 目标函数：|| Tsum - R2 ||_F^2 =====
    minimize( square_pos( norm(Tsum - R2, 'fro') ) )

    subject to
        % ===== (20c) per-antenna power 约束 =====
        % diag( sum_i T_i ) = P0/N * 1_N
        diag(Tsum) == (P0/N) * ones(N,1);

        % ===== (20d) PSD 约束：T_i ⪰ 0 =====
        T1 >= 0;
        T2 >= 0;
        T3 >= 0;
        T4 >= 0;

        % ===== (20b) SINR 约束：γ_i >= gamma，γ_i 用 (7) =====
        % γ_i = tr(HiHiH{i} * T_i) / ( sum_{k≠i} tr(HiHiH{i} * T_k) + N0 )

        % 用户 1：
        trace(HiHiH{1} * T1) >= gamma * ( ...
            trace(HiHiH{1} * T2) + ...
            trace(HiHiH{1} * T3) + ...
            trace(HiHiH{1} * T4) + N0 );

        % 用户 2：
        trace(HiHiH{2} * T2) >= gamma * ( ...
            trace(HiHiH{2} * T1) + ...
            trace(HiHiH{2} * T3) + ...
            trace(HiHiH{2} * T4) + N0 );

        % 用户 3：
        trace(HiHiH{3} * T3) >= gamma * ( ...
            trace(HiHiH{3} * T1) + ...
            trace(HiHiH{3} * T2) + ...
            trace(HiHiH{3} * T4) + N0 );

        % 用户 4：
        trace(HiHiH{4} * T4) >= gamma * ( ...
            trace(HiHiH{4} * T1) + ...
            trace(HiHiH{4} * T2) + ...
            trace(HiHiH{4} * T3) + N0 );
cvx_end

% SDR 得到的解：T_i 可能不是 rank-1，但满足 (20a)(20b)(20c)(20d 的 PSD 部分)
C_radcom = Tsum;   % 通信协方差矩阵，同时作为雷达 probing 协方差

% RadCom 情况下的方向图：P_radcom(theta) = a^H C_radcom a
P_radcom = zeros(M,1);
for m = 1:M
    P_radcom(m) = real(trace(C_radcom * Sfull{m}));
end

%% ============= 归一化（每条曲线用自己的最大值）并绘图 ===================

% Ideal：用自身最大值归一化
norm_ideal       = max(P_ideal);
P_ideal_n        = P_ideal / (norm_ideal + eps);

% Radar-Only（Shared）：用雷达-only 方向图自身的最大值归一化
%norm_radar_only  = max(P_radar_only);
norm_radar_only=5;
P_radar_only_n   = P_radar_only / (norm_radar_only + eps);

% RadCom（Shared）：用 RadCom 方向图自身的最大值归一化
%norm_radcom      = max(P_radcom);
norm_radcom=5;
P_radcom_n       = P_radcom / (norm_radcom + eps);

% 线性刻度
figure;
plot(theta_deg, P_ideal_n, 'k--','LineWidth',1.5); hold on;
plot(theta_deg, P_radar_only_n,'b-','LineWidth',1.5);
plot(theta_deg, P_radcom_n,'r-','LineWidth',1.5);
grid on; xlim([-90 90]);
xlabel('Angle (Degree)');
ylabel('Normalized Beampattern');
legend('Ideal','Radar-Only (Shared)','RadCom (Shared)','Location','Best');
title('Fig.3(b) Shared Deployment: Multi-beam Beampatterns (self-normalized)');
