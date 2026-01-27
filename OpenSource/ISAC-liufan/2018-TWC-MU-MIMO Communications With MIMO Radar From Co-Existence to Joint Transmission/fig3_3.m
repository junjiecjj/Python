% =============================================================
clear; clc; close all;

%% ============= System parameters =============
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
main_bw_deg   = 3;           % half mainlobe width半波长


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

%% ====== Normalization for Separated Deployment (Fig.3(a) style) ======

% 1) 线性功率（已经是 a^H C a 形式），先取绝对值保证非负
P_radar_sep  = abs(beamp_sep_rad);         % Radar-Only
P_ideal_sep  = abs(alpha_sep * P_tilde);   % Ideal (用 alpha_sep * P_tilde 对齐论文)
% P_radcom_sep = abs(beamp_radcom_sep);   % RadCom（可选）

% 2) 统一归一化因子：在同一个 subplot 中，所有曲线共用同一个最大值

%Pmax_sep = max( [P_radar_sep; P_ideal_sep] );
Pmax_sep = 50/14;
% 如果有 RadCom，就写：Pmax_sep = max( [P_radar_sep; P_ideal_sep; P_radcom_sep] );

P_radar_sep_n = P_radar_sep / Pmax_sep;    % 归一化 Radar-Only
P_ideal_sep_n = P_ideal_sep / Pmax_sep;    % 归一化 Ideal
% P_radcom_sep_n = P_radcom_sep / Pmax_sep; % 归一化 RadCom（有时再用）

%% --------------------- 画图（分离式子图） ------------------------
figure;
plot(theta_deg, P_ideal_sep_n, 'k:','LineWidth',1.5); hold on;      % Ideal（黑色点虚线）
plot(theta_deg, P_radar_sep_n,'b--','LineWidth',1.8);              % Radar-Only（蓝色虚线）


grid on;
xlabel('Angle (Degree)');
ylabel('Normalized Beampattern');
title('Separated Deployment');
%%legend('Ideal','Radar-Only'/*,'RadCom'*/, 'Location','Best');
xlim([-90 90]);
ylim([0 5]);          % 跟论文 Fig.3(a) 一样给个 0~5 的刻度
