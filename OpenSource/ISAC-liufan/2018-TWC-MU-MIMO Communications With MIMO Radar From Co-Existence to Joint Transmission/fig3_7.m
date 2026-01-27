% =============================================================
% Fig.3(a) - Separated Deployment (Radar-only & RadCom)
% MU-MIMO Communications with MIMO Radar:
%   From Co-existence to Joint Transmission (Liu et al., TSP 2018)
%
% 实现论文 Eq.(12)、Eq.(19)、Eq.(14)、Eq.(15)
% =============================================================
clear; clc; close all;

%% ================== System Parameters ======================
P0_dBm = 20; 
P0     = 10^(P0_dBm/10);
N      = 20;        % total antennas
NR     = 14;        % radar antennas
NC     = N - NR;    % comm antennas
K      = 4;         % users

N0_dBm = 0;
N0     = 10^(N0_dBm/10);

lambda = 1;
d      = 0.5 * lambda;

gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

PR = P0/2;
PC = P0/2;

rng(1);

%% ================= Angle grid & Steering ====================
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m))/lambda);
end

A1 = A_full(1:NR ,:);    % radar
A2 = A_full(NR+1:end,:); % comm

% Precompute S matrices
S1 = cell(M,1);
S2 = cell(M,1);
Sfull = cell(M,1);
for m = 1:M
    a1m = A1(:,m); 
    a2m = A2(:,m);
    afm = A_full(:,m);
    S1{m} = a1m * a1m';
    S2{m} = a2m * a2m';
    Sfull{m} = afm * afm';
end

%% ================== Desired Multi-beam Pattern ================
beam_dirs = [-60 -36 0 36 60];
bw = 4;  % ±4 degrees (as in the paper)

P_tilde = zeros(M,1);
for b = 1:numel(beam_dirs)
    idx = abs(theta_deg - beam_dirs(b)) <= bw;
    P_tilde(idx) = 2;
end

%% ================== Rayleigh Channels ==========================
H = (randn(N,K)+1j*randn(N,K))/sqrt(2);
F = H(1:NR ,:);
G = H(NR+1:end,:);

FiFiH = cell(K,1);
GiGiH = cell(K,1);
for k = 1:K
    FiFiH{k} = F(:,k)*F(:,k)';
    GiGiH{k} = G(:,k)*G(:,k)';
end

%% =============================================================
% Part 1 — Radar-only (Separated) : Solve Eq.(12)
% =============================================================

cvx_begin sdp quiet
    variable R1(NR,NR) hermitian semidefinite
    variable alpha_sep
    expression beamp(M)

    for m = 1:M
        beamp(m) = real(trace(R1 * S1{m}));
    end

    minimize( sum_square( alpha_sep * P_tilde - beamp ) )

    subject to
        diag(R1) == (PR/NR) * ones(NR,1);
        alpha_sep >= 0;

        % Zero interference to users
        for k = 1:K
            real(trace(R1 * FiFiH{k})) == 0;
        end
cvx_end

P_radar_sep = beamp;

%% =============================================================
% Part 2 — RadCom (Separated) : Solve Eq.(19)
% =============================================================

cvx_begin sdp quiet
    variable W1(NC,NC) hermitian semidefinite
    variable W2(NC,NC) hermitian semidefinite
    variable W3(NC,NC) hermitian semidefinite
    variable W4(NC,NC) hermitian semidefinite
    variable sigma_sep

    expression sumW(NC,NC)
    sumW = W1 + W2 + W3 + W4;

    C_comm  = A2' * sumW * A2;
    C_radar = A1' * R1 * A1;

    diff_vec = real(diag(C_comm - sigma_sep * C_radar));
    minimize( sum_square(diff_vec) )

    subject to
        % SINR constraints
        for k = 1:K
            num = real(trace(eval(sprintf('W%d',k)) * GiGiH{k}));
            den = N0;

            for j = 1:K
                if j ~= k
                    den = den + real(trace(eval(sprintf('W%d',j)) * GiGiH{k}));
                end
            end
            
            den = den + real(trace(R1 * FiFiH{k}));
            num >= gamma * den;
        end

        trace(sumW) <= PC;
        sigma_sep >= 0;
cvx_end

Wsum = full(W1+W2+W3+W4);

% Compute RadCom beampattern (Eq.15)
P_radcom_sep = zeros(M,1);
for m = 1:M
    P_radcom_sep(m) = real(trace( blkdiag(R1,Wsum) * Sfull{m} ));
end

%% ================= Normalization (as in Fig.3(a)) ======================

norm_ideal = max(alpha_sep * P_tilde);
norm_radar = max(P_radar_sep);
norm_radc  = max(P_radcom_sep);

P_ideal_n  = (alpha_sep * P_tilde)  / norm_ideal;
P_radar_n  = P_radar_sep            / norm_radar;
P_radc_n   = P_radcom_sep           / norm_radc;

%% ========================= Plot ======================================

figure;
plot(theta_deg, P_ideal_n,'k:','LineWidth',1.6); hold on;
plot(theta_deg, P_radar_n,'b--','LineWidth',1.8);
plot(theta_deg, P_radc_n,'r','LineWidth',1.8);

xlabel('Angle'); ylabel('Normalized Beampattern');
title('Fig.3(a) Separated Deployment: Radar-only & RadCom');
legend('Ideal','Radar-only','RadCom','Location','Best');
grid on;
xlim([-90 90]);
ylim([0 1.1]);
