% =============================================================
% Fig.4(b) - Shared Deployment (3 dB Beampattern, Array Gain in dBi)
% MU-MIMO Communications with MIMO Radar: From Co-existence to Joint Transmission
%
% Radar-only (Problem 10): strict 3 dB equality, sidelobe set = outside [-5,5]
% RadCom     (Problem 20): Frobenius matrix matching  ||Wsum - sigma*R2||_F^2
%
% Plot: Array Gain (dBi): G(theta)= P(theta)/(P0/N), dBi=10log10(G)
% =============================================================
clear; clc; close all;

%% ============= System parameters =============
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear total power
N      = 20;                 % total antennas (shared)
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);

lambda = 1;
d      = 0.5*lambda;

gamma_dB = 10;               % SINR target
gamma    = 10^(gamma_dB/10);

rng(1);

%% ============= Angle grid and steering matrices =============
theta_deg = -90:0.5:90;      % keep 0.5 deg for smooth Fig.4
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

A = zeros(N,M);
for m = 1:M
    A(:,m) = exp(1j*2*pi*(d/lambda)*(0:N-1)'*sin(theta_rad(m)));
end

%% ============= Random user channels (Rayleigh) ==============
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);
HiHiH = cell(K,1);
for i = 1:K
    hi = H(:,i);
    HiHiH{i} = hi*hi';
end

%% =============================================================
% Part 1: Radar-Only (Shared) via Problem (10) - strict 3 dB
% =============================================================
theta0 = 0;
bw_3dB = 10;
theta1 = theta0 - bw_3dB/2;   % -5
theta2 = theta0 + bw_3dB/2;   % +5

[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

a0 = A(:,idx0);
a1 = A(:,idx1);
a2 = A(:,idx2);

% ---- sidelobe set for DESIGN: no guard (paper-like) ----
idx_sidelobe_design = find( (theta_deg < theta1) | (theta_deg > theta2) );

fprintf('Solving Radar-only (Shared, Problem 10) via CVX...\n');
cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable t_sh
    minimize( -t_sh )
    subject to
        % per-antenna equality power
        diag(R2) == (P0/N)*ones(N,1);

        main0 = real(a0' * R2 * a0);
        % strict 3 dB equality
        real(a1' * R2 * a1) == 0.5*main0;
        real(a2' * R2 * a2) == 0.5*main0;

        % sidelobe suppression
        for mm = idx_sidelobe_design
            am = A(:,mm);
            real(main0 - am' * R2 * am) >= t_sh;
        end
cvx_end

if ~(strcmpi(cvx_status,'Solved') || strcmpi(cvx_status,'Inaccurate/Solved'))
    error('Radar-only (Problem 10) failed: %s', cvx_status);
end

% Radar-only beampattern power P(theta)=a^H R2 a
P_radar = real(diag(A' * R2 * A));

%% =============================================================
% Part 2: RadCom (Shared) via Problem (20) - Frobenius matching
%         minimize ||Wsum - sigma*R2||_F^2
% =============================================================
fprintf('Solving RadCom (Shared, Problem 20) via CVX (Frobenius match)...\n');

cvx_begin sdp quiet
    variable W1(N,N) hermitian semidefinite
    variable W2(N,N) hermitian semidefinite
    variable W3(N,N) hermitian semidefinite
    variable W4(N,N) hermitian semidefinite
    variable sig_sh

    expression Wsum(N,N)
    Wsum = W1 + W2 + W3 + W4;

    minimize( square_pos( norm(Wsum - sig_sh*R2, 'fro') ) )

    subject to
        % per-antenna equality (shared deployment)
        diag(Wsum) == (P0/N)*ones(N,1);

        % SINR constraints
        real(trace(HiHiH{1}*W1)) >= gamma*( real(trace(HiHiH{1}*W2))+real(trace(HiHiH{1}*W3))+real(trace(HiHiH{1}*W4)) + N0 );
        real(trace(HiHiH{2}*W2)) >= gamma*( real(trace(HiHiH{2}*W1))+real(trace(HiHiH{2}*W3))+real(trace(HiHiH{2}*W4)) + N0 );
        real(trace(HiHiH{3}*W3)) >= gamma*( real(trace(HiHiH{3}*W1))+real(trace(HiHiH{3}*W2))+real(trace(HiHiH{3}*W4)) + N0 );
        real(trace(HiHiH{4}*W4)) >= gamma*( real(trace(HiHiH{4}*W1))+real(trace(HiHiH{4}*W2))+real(trace(HiHiH{4}*W3)) + N0 );

        sig_sh >= 0;
cvx_end

if ~(strcmpi(cvx_status,'Solved') || strcmpi(cvx_status,'Inaccurate/Solved'))
    error('RadCom (Problem 20) failed: %s', cvx_status);
end

% RadCom beampattern power P(theta)=a^H Wsum a
P_radcom = real(diag(A' * Wsum * A));

%% =============================================================
% Part 3: Array Gain (dBi) and plotting (paper Fig.4 style)
% =============================================================
G_radar_lin  = P_radar  / (P0/N);
G_radcom_lin = P_radcom / (P0/N);

G_radar_dBi  = 10*log10(G_radar_lin  + 1e-12);
G_radcom_dBi = 10*log10(G_radcom_lin + 1e-12);

% ---- PSLR evaluation (guard ONLY for evaluation, not for design) ----
guard_eval = 1;  % use 1 if you used it to match paper for separated
idx_main   = (theta_deg >= theta1) & (theta_deg <= theta2);
idx_side   = (theta_deg <= theta1-guard_eval) | (theta_deg >= theta2+guard_eval);

PSLR_radar  = 10*log10( max(G_radar_lin(idx_main))  / max(G_radar_lin(idx_side)) );
PSLR_radcom = 10*log10( max(G_radcom_lin(idx_main)) / max(G_radcom_lin(idx_side)) );
fprintf('PSLR eval (guard=%d): Radar-only=%.2f dB, RadCom=%.2f dB\n', guard_eval, PSLR_radar, PSLR_radcom);

% ---- Plot ----
figure;
plot(theta_deg, G_radar_dBi,  'b--','LineWidth',1.8); hold on;
plot(theta_deg, G_radcom_dBi, 'r','LineWidth',1.8);
grid on;
xlabel('Angle (Degree)');
ylabel('Array Gain (dBi)');
title('Shared Deployment - 3 dB Beampattern (Fig.4(b) paper-like)');
legend('Radar-Only','RadCom','Location','Best');
xlim([-90 90]);
ylim([-20 10]);
