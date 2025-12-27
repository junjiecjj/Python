% =============================================================
% Fig.4(b) - Shared Deployment (3 dB Beampattern)
% MU-MIMO Communications with MIMO Radar: From Co-existence to Joint Transmission
%
% Step 1) Radar-only (shared): solve Problem (10) to get R2 (NxN)
% Step 2) RadCom (shared):     solve SDR of Problem (20) to get {Tk} (NxN)
% Step 3) Plot beampattern using Array Gain (dBi):
%         G(theta) = P(theta) / (P0/N),  dBi = 10log10(G)
% =============================================================
clear; clc; close all;

%% =================== System parameters ===================
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear
N      = 20;                 % total antennas at BS
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);     % linear

lambda = 1;                  % normalized wavelength
d      = 0.5*lambda;         % half-wavelength spacing

% SINR target for all users (10 dB)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

rng(1);                      % reproducibility

%% =================== Angle grid & steering ===================
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

% Full N-antenna ULA steering matrix A (N x M)
A = zeros(N,M);
for m = 1:M
    A(:,m) = exp(1j*2*pi*(d/lambda)*(0:N-1)' * sin(theta_rad(m)));
end

%% =================== User channels (Rayleigh) ===================
% h_i ~ CN(0, I), size N x 1
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

HiHiH = cell(K,1);           % hi hi^H (N x N)
for i = 1:K
    hi       = H(:,i);
    HiHiH{i} = hi * hi';
end

% =============================================================
% Part 1: Radar-only (Shared) via Problem (10) - 3 dB beam
% =============================================================

% ---- Define main-beam and sidelobe regions ----
theta0   = 0;                 % main-beam center (deg)
bw_3dB   = 10;                % 3 dB beamwidth (deg)
theta1   = theta0 - bw_3dB/2; % -5 deg
theta2   = theta0 + bw_3dB/2; % +5 deg

[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

idx_sidelobe = find( (theta_deg < theta1) | (theta_deg > theta2) );

a0 = A(:,idx0);
a1 = A(:,idx1);
a2 = A(:,idx2);

fprintf('Solving Radar-only (Shared, Problem 10) via CVX...\n');
cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable t_sh

    minimize( -t_sh )   % maximize t_sh
    subject to
        % Per-antenna equal power (paper uses equality):
        diag(R2) == (P0/N) * ones(N,1);

        % 3 dB constraints (half-power at theta1/theta2):
        main0 = real( a0' * R2 * a0 );
        left  = real( a1' * R2 * a1 );
        right = real( a2' * R2 * a2 );
        left  == 0.5 * main0;
        right == 0.5 * main0;

        % Sidelobe suppression: main0 - sidelobe >= t_sh, for θ in Ω
        for mm = idx_sidelobe
            am    = A(:,mm);
            sidel = real( am' * R2 * am );
            main0 - sidel >= t_sh;
        end
cvx_end

% Radar-only beampattern (shared)
beamp_radar = zeros(M,1);
for m = 1:M
    am = A(:,m);
    beamp_radar(m) = real( am' * R2 * am );
end

% =============================================================
% Part 2: RadCom (Shared) via SDR of Problem (20)
%   minimize || sum_k T_k - R2 ||_F^2
%   s.t. SINR_i >= gamma, diag(sum_k T_k) = P0/N, T_k >= 0 (PSD)
% =============================================================

fprintf('Solving RadCom (Shared, Problem 20 - SDR) via CVX...\n');

cvx_begin sdp quiet
    variable T1(N,N) hermitian semidefinite
    variable T2(N,N) hermitian semidefinite
    variable T3(N,N) hermitian semidefinite
    variable T4(N,N) hermitian semidefinite

    expression sumT(N,N)
    sumT = T1 + T2 + T3 + T4;

    % Objective: Frobenius norm squared
    minimize( square_pos( norm(sumT - R2, 'fro') ) )

    subject to
        % Per-antenna equal power constraint (paper uses equality):
        diag(sumT) == (P0/N) * ones(N,1);

        % SINR constraints (shared SINR):
        % tr(HiHiH{i}*Ti) >= gamma*( tr(HiHiH{i}*(sumT - Ti)) + N0 )
        % where tr(HiHiH{i}*T) = hi^H T hi
        for i = 1:K
            if i == 1
                Ti = T1;
            elseif i == 2
                Ti = T2;
            elseif i == 3
                Ti = T3;
            else
                Ti = T4;
            end

            num = real( trace( Ti * HiHiH{i} ) );
            den = real( trace( (sumT - Ti) * HiHiH{i} ) ) + N0;

            num >= gamma * den;
        end
cvx_end

% Shared RadCom beampattern
beamp_radcom = zeros(M,1);
for m = 1:M
    am = A(:,m);
    beamp_radcom(m) = real( am' * sumT * am );
end

% =============================================================
% Part 3: Array Gain normalization -> dBi, PSLR, Plot (Fig.4(b))
% =============================================================

% Array Gain definition: G(theta) = P(theta) / (P0/N)
G_radar_lin  = beamp_radar  / (P0/N);
G_radcom_lin = beamp_radcom / (P0/N);

G_radar_dBi  = 10*log10(G_radar_lin  + 1e-12);
G_radcom_dBi = 10*log10(G_radcom_lin + 1e-12);

% PSLR (mainlobe region [theta1, theta2])
idx_main = (theta_deg >= theta1) & (theta_deg <= theta2);
idx_side = ~idx_main;

PSLR_radar_dB  = max(G_radar_dBi(idx_main))  - max(G_radar_dBi(idx_side));
PSLR_radcom_dB = max(G_radcom_dBi(idx_main)) - max(G_radcom_dBi(idx_side));
fprintf('PSLR (Radar-only, Shared 3dB) : %.2f dB\n', PSLR_radar_dB);
fprintf('PSLR (RadCom, Shared 3dB)     : %.2f dB\n', PSLR_radcom_dB);

% Plot (paper-style)
figure;
plot(theta_deg, G_radar_dBi,  'b--','LineWidth',1.8); hold on;
plot(theta_deg, G_radcom_dBi, 'r','LineWidth',1.8);
grid on;
xlabel('Angle (Degree)');
ylabel('Array Gain (dBi)');
title('Shared Deployment - 3 dB Beampattern (Fig.4(b))');
legend('Radar-Only','RadCom (Shared, SDR)','Location','Best');
xlim([-90 90]);
ylim([-20 10]);  % 按论文视觉风格；你也可注释掉让其自动
