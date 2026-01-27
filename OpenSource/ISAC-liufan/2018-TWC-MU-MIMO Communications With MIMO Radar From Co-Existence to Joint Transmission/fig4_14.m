% =============================================================
% Fig.4(b) - Shared Deployment (3 dB Beampattern, Array Gain)
% UPDATED (same "PSLR-improving" settings as Fig.4(a)):
%   (1) angle grid: 1 deg
%   (2) sidelobe set with guard band (guard=1 recommended)
%   (3) relaxed 3dB constraints: <= 0.5 * main0
% Output: Array Gain (dBi), NOT self-normalized beampattern
% =============================================================
clear; clc; close all;

%% ============= System parameters =============
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear total power
N      = 20;                 % total antennas (shared)
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);     % linear

lambda = 1;                  % normalized wavelength
d      = 0.5*lambda;         % half-wavelength spacing

% SINR target for all users (10 dB)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

rng(1);

%% ============= Angle grid and steering (UPDATED: 1 deg) =============
theta_deg = -90:1:90;              % <<< UPDATED
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

A = zeros(N,M);
for m = 1:M
    A(:,m) = exp(1j*2*pi*(d/lambda)*(0:N-1)'*sin(theta_rad(m)));
end

% 3 dB spec
theta0 = 0;
bw_3dB = 10;
theta1 = theta0 - bw_3dB/2;   % -5
theta2 = theta0 + bw_3dB/2;   % +5

[~, idx0] = min(abs(theta_deg-theta0));
[~, idx1] = min(abs(theta_deg-theta1));
[~, idx2] = min(abs(theta_deg-theta2));

a0 = A(:,idx0);
a1 = A(:,idx1);
a2 = A(:,idx2);

% UPDATED: sidelobe set with guard band (use guard=1 to match paper)
guard = 1;
idx_sidelobe = find( (theta_deg <= theta1-guard) | (theta_deg >= theta2+guard) );

%% ============= Random user channels (Rayleigh) ==============
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

HiHiH = cell(K,1);   % N x N
for i = 1:K
    hi = H(:,i);
    HiHiH{i} = hi*hi';
end

%% =============================================================
% Part 1: Radar-Only (Shared) via Problem (10) - 3 dB
% =============================================================
fprintf('Solving radar-only 3dB (shared, Problem 10) via CVX...\n');
cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable t_sh
    minimize( -t_sh )
    subject to
        % per-antenna power equality
        diag(R2) == (P0/N) * ones(N,1);

        main0 = real( a0' * R2 * a0 );
        left  = real( a1' * R2 * a1 );
        right = real( a2' * R2 * a2 );

        % UPDATED: relaxed 3dB constraints
        left  <= 0.5 * main0;
        right <= 0.5 * main0;

        % sidelobe suppression
        for mm = idx_sidelobe
            am = A(:,mm);
            sidel = real( am' * R2 * am );
            main0 - sidel >= t_sh;
        end
cvx_end

%% =============================================================
% Part 2: RadCom (Shared) via Problem (20) - SDR
%   minimize || Wsum - sig*R2 ||_F^2
%   s.t. diag(Wsum)=P0/N, SINR>=gamma, Wi PSD, sig>=0
% =============================================================
fprintf('Solving RadCom 3dB (shared, Problem 20) via CVX...\n');
cvx_begin sdp quiet
    variable W(N,N,K) hermitian semidefinite
    variable sig_sh

    expression Wsum(N,N)
    Wsum = 0;
    for k = 1:K
        Wsum = Wsum + W(:,:,k);
    end

    minimize( square_pos(norm(Wsum - sig_sh*R2,'fro')) )

    subject to
        % per-antenna power equality (paper shared constraint)
        diag(Wsum) == (P0/N) * ones(N,1);

        sig_sh >= 0;

        % SINR constraints
        for i = 1:K
            num = real(trace(HiHiH{i} * W(:,:,i)));

            interf = 0;
            for k = 1:K
                if k ~= i
                    interf = interf + real(trace(HiHiH{i} * W(:,:,k)));
                end
            end

            num >= gamma * (interf + N0);
        end
cvx_end

%% =============================================================
% Part 3: Beampattern -> Array Gain (dBi) and plot Fig.4(b)
%   P(theta)=a^H C a,   Gain = P(theta)/(P0/N)
% =============================================================
P_radar = zeros(M,1);
P_radcom= zeros(M,1);

for m = 1:M
    am = A(:,m);
    P_radar(m)  = real(am' * R2   * am);
    P_radcom(m) = real(am' * Wsum * am);
end

G_radar_lin  = P_radar  / (P0/N);
G_radcom_lin = P_radcom / (P0/N);

G_radar_dBi  = 10*log10(G_radar_lin  + 1e-12);
G_radcom_dBi = 10*log10(G_radcom_lin + 1e-12);

% optional: print PSLR using the same guard definition
idx_main = (theta_deg >= theta1) & (theta_deg <= theta2);
idx_side = (theta_deg <= theta1-guard) | (theta_deg >= theta2+guard);

PSLR_radar  = 10*log10( max(G_radar_lin(idx_main))  / max(G_radar_lin(idx_side)) );
PSLR_radcom = 10*log10( max(G_radcom_lin(idx_main)) / max(G_radcom_lin(idx_side)) );
fprintf('Shared: PSLR radar-only = %.2f dB, PSLR RadCom = %.2f dB (guard=%d)\n', ...
    PSLR_radar, PSLR_radcom, guard);

%% ---------------- Plot: Fig.4(b) style ----------------
figure;
plot(theta_deg, G_radar_dBi,  'b--','LineWidth',1.8); hold on;
plot(theta_deg, G_radcom_dBi, 'r','LineWidth',1.8);
grid on;
xlabel('Angle (Degree)');
ylabel('Array Gain (dBi)');
title('Shared Deployment - 3 dB Beampattern (UPDATED Fig.4(b))');
legend('Radar-Only','RadCom','Location','Best');
xlim([-90 90]);
ylim([-20 10]);   % same visual range as Fig.4(a)

