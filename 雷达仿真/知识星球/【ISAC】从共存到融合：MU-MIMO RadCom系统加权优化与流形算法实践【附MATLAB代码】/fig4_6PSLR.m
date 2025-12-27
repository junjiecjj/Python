% =============================================================
%  Fig.4(b) - Shared Deployment (3 dB Beampattern, compute PSLR)
% =============================================================
clear; clc; close all;

%% ============= System parameters =============
P0_dBm = 20;
P0     = 10^(P0_dBm/10);
N      = 20;      
K      = 4;

N0_dBm = 0;
N0     = 10^(N0_dBm/10);

lambda = 1;
d      = 0.5*lambda;

gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

rng(1);

%% ============= Angle grid and steering matrices =============
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m)));
end

% Sfull{m} = a_m a_m^H
Sfull = cell(M,1);
for m = 1:M
    Sfull{m} = A_full(:,m) * A_full(:,m)';
end

%% ============= Random user channels ==============
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

HiHiH = cell(K,1);
for i = 1:K
    HiHiH{i} = H(:,i) * H(:,i)';
end

%% =============================================================
%  Part 1: Radar-Only (Shared) — CVX solve Problem (10)
% =============================================================
theta0   = 0;
bw_3dB   = 10;
theta1   = theta0 - bw_3dB/2;    
theta2   = theta0 + bw_3dB/2;    

[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

idx_sidelobe = find( (theta_deg < theta1) | (theta_deg > theta2) );

fprintf('Solving Radar-only (Problem 10)...\n');

cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable t0

    minimize( -t0 )

    subject to
        diag(R2) == (P0/N);

        P0_main  = real(trace(R2 * Sfull{idx0}));
        P_left   = real(trace(R2 * Sfull{idx1}));
        P_right  = real(trace(R2 * Sfull{idx2}));

        P_left  == 0.5 * P0_main;
        P_right == 0.5 * P0_main;

        for mm = idx_sidelobe
            P_sl = real(trace(R2 * Sfull{mm}));
            P0_main - P_sl >= t0;
        end
cvx_end

% Radar-only beampattern
P_radar = zeros(M,1);
for m = 1:M
    P_radar(m) = real(trace(R2 * Sfull{m}));
end

%% =============================================================
%  Part 2: RadCom (Shared) — CVX solve Problem (20)
% =============================================================
fprintf('Solving RadCom (Problem 20)...\n');

cvx_begin sdp quiet
    variable W1(N,N) hermitian semidefinite
    variable W2(N,N) hermitian semidefinite
    variable W3(N,N) hermitian semidefinite
    variable W4(N,N) hermitian semidefinite
    variable sig

    expression Wsum(N,N)
    Wsum = W1 + W2 + W3 + W4;

    % beampattern matching objective
    expression diff_vec(M,1)
    for m = 1:M
        diff_vec(m) = real( trace(Wsum * Sfull{m}) ...
                         - sig * trace(R2 * Sfull{m}) );
    end
    minimize( sum_square(diff_vec) )

    subject to
        diag(Wsum) == (P0/N);

        W1 >= 0; W2 >= 0; W3 >= 0; W4 >= 0;
        sig >= 0;

        % SINR constraints
        trace(HiHiH{1}*W1) >= gamma * ( trace(HiHiH{1}*W2) + trace(HiHiH{1}*W3) + trace(HiHiH{1}*W4) + N0 );
        trace(HiHiH{2}*W2) >= gamma * ( trace(HiHiH{2}*W1) + trace(HiHiH{2}*W3) + trace(HiHiH{2}*W4) + N0 );
        trace(HiHiH{3}*W3) >= gamma * ( trace(HiHiH{3}*W1) + trace(HiHiH{3}*W2) + trace(HiHiH{3}*W4) + N0 );
        trace(HiHiH{4}*W4) >= gamma * ( trace(HiHiH{4}*W1) + trace(HiHiH{4}*W2) + trace(HiHiH{4}*W3) + N0 );
cvx_end

% RadCom beampattern
P_radcom = zeros(M,1);
for m = 1:M
    P_radcom(m) = real(trace(Wsum * Sfull{m}));
end

%% =============================================================
%  Compute PSLR for radar-only and RadCom
% =============================================================

% 主瓣功率
P0_main = P_radar(idx0);

% 旁瓣最大
P_sidelobe_radar = max(P_radar(idx_sidelobe));

% PSLR (Radar-only)
PSLR_radar = 10*log10(P_sidelobe_radar / P0_main);

% RadCom PSLR
P0_main_rc = P_radcom(idx0);
P_sidelobe_rc = max(P_radcom(idx_sidelobe));
PSLR_radcom = 10*log10(P_sidelobe_rc / P0_main_rc);

fprintf('\n===== PSLR Results (in dB, smaller = better) =====\n');
fprintf('Radar-only PSLR = %.2f dB\n', PSLR_radar);
fprintf('RadCom     PSLR = %.2f dB\n', PSLR_radcom);


%% =============================================================
%  Normalized plots (same style as Fig.4(b))
% =============================================================

P_radar_n  = P_radar  / max(P_radar);
P_radcom_n = P_radcom / max(P_radcom);

figure;
plot(theta_deg,10*log10(P_radar_n),'b--','LineWidth',1.4); hold on;
plot(theta_deg,10*log10(P_radcom_n),'r','LineWidth',1.4);
grid on; xlim([-90 90]);
xlabel('Angle (deg)');
ylabel('Normalized Beampattern (dB)');
legend('Radar-only 3dB','RadCom 3dB');
title('Fig.4(b) Shared Deployment (3 dB Beampattern) + PSLR');
