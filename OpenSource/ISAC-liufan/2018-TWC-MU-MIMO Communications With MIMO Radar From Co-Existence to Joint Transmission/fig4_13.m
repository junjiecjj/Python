% =============================================================
% Fig.4(a) - Separated Deployment (3 dB Beampattern, Array Gain)
% UPDATED to improve PSLR:
%   (1) angle grid: 1 deg
%   (2) sidelobe set with guard band
%   (3) relaxed 3dB constraints: <= 0.5 * main0 (not equality)
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

rng(1);                      % reproducibility

%% ============= Angle grid and steering matrices (UPDATED: 1 deg) =============
theta_deg = -90:1:90;                % <<< UPDATED
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m))/lambda);
end

A1 = A_full(1:NR ,:);        % radar subarray
A2 = A_full(NR+1:end,:);     % comm  subarray

%% ============= Random user channels (Rayleigh) ==============
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);  % N x K

F = H(1:NR ,:);              % radar -> users
G = H(NR+1:end,:);           % comm  -> users

GiGiH = cell(K,1);
FiFiH = cell(K,1);
for i = 1:K
    GiGiH{i} = G(:,i) * G(:,i)';
    FiFiH{i} = F(:,i) * F(:,i)';
end

%% =============================================================
% Part 1: Radar-Only (Separated) via Problem (13) - 3 dB beam
% =============================================================
theta0 = 0;                   % main-beam center
bw_3dB = 10;                  % 3 dB beamwidth
theta1 = theta0 - bw_3dB/2;   % -5
theta2 = theta0 + bw_3dB/2;   % +5

[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

a1_0 = A1(:,idx0);
a1_1 = A1(:,idx1);
a1_2 = A1(:,idx2);

% --------- UPDATED: sidelobe set with guard band ----------
guard = 1;  
idx_sidelobe = find( (theta_deg <= theta1-guard) | (theta_deg >= theta2+guard) );

fprintf('Solving radar-only 3dB (separated, Problem 13) via CVX...\n');
cvx_begin sdp quiet
    variable R1(NR,NR) hermitian semidefinite
    variable t_sep
    minimize( -t_sep )
    subject to
        % (i) per-antenna power on radar array
        diag(R1) == (PR/NR) * ones(NR,1);

        % (ii) UPDATED: relaxed 3dB constraints (<= half-power)
        main0 = real( a1_0' * R1 * a1_0 );
        left  = real( a1_1' * R1 * a1_1 );
        right = real( a1_2' * R1 * a1_2 );
        left  <= 0.5 * main0;
        right <= 0.5 * main0;

        % (iii) sidelobe suppression
        for mm = idx_sidelobe
            a1m   = A1(:,mm);
            sidel = real( a1m' * R1 * a1m );
            main0 - sidel >= t_sep;
        end

        % (iv) ZF radar->users
        for i = 1:K
            real( trace( R1 * FiFiH{i} ) ) == 0;
        end
cvx_end

% Full array covariance for radar-only
C_radar_sep = blkdiag(R1, zeros(NC,NC));

beamp_radar_sep = zeros(M,1);
for m = 1:M
    af = A_full(:,m);
    beamp_radar_sep(m) = real( af' * C_radar_sep * af );
end

%% =============================================================
% Part 2: RadCom (Separated, Problem 19)
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

    diff_vec = real( diag( C_comm - sig_sep * C_radar_A1 ) );
    minimize( sum_square( diff_vec ) )

    subject to
        % SINR constraints
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

        trace(sumW) <= PC;
        sig_sep >= 0;
cvx_end

C_radcom_sep = blkdiag(R1, W1+W2+W3+W4);

beamp_radcom_sep = zeros(M,1);
for m = 1:M
    af = A_full(:,m);
    beamp_radcom_sep(m) = real( af' * C_radcom_sep * af );
end

%% =============================================================
% Part 3: Array Gain -> dBi (as in paper Fig.4)
% =============================================================
% Radar-only: average per active radar antenna = PR/NR
G_radar_lin  = beamp_radar_sep / (PR/NR);

% RadCom: paper plots array gain using average per antenna = P0/N
G_radcom_lin = beamp_radcom_sep / (P0/N);

G_radar_dBi  = 10*log10(G_radar_lin  + 1e-12);
G_radcom_dBi = 10*log10(G_radcom_lin + 1e-12);

% Optional: print PSLR (using main=[-5,5], sidelobe with same guard set)
idx_main = (theta_deg >= theta1) & (theta_deg <= theta2);
idx_side = (theta_deg <= theta1-guard) | (theta_deg >= theta2+guard);
PSLR_radar = 10*log10( max(G_radar_lin(idx_main)) / max(G_radar_lin(idx_side)) );
PSLR_radcom= 10*log10( max(G_radcom_lin(idx_main)) / max(G_radcom_lin(idx_side)) );
fprintf('PSLR (Radar-only) = %.2f dB,  PSLR (RadCom) = %.2f dB  (guard=%d deg)\n', ...
    PSLR_radar, PSLR_radcom, guard);

%% ---------------- Plot Fig.4(a) ----------------
figure;
plot(theta_deg, G_radar_dBi,  'b--','LineWidth',1.8); hold on;
plot(theta_deg, G_radcom_dBi, 'r','LineWidth',1.8);
grid on;
xlabel('Angle (Degree)');
ylabel('Array Gain (dBi)');
title('Separated Deployment - 3 dB Beampattern (UPDATED Fig.4(a))');
legend('Radar-Only','RadCom','Location','Best');
xlim([-90 90]);
ylim([-20 10]);
