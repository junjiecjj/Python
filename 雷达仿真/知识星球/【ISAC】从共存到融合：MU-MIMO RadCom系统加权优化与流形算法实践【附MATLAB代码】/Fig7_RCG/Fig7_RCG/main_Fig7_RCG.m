% =====================================================================
%  main_Fig7_RCG.m
%
%  Reproduce Fig.7 (solid lines): Max-RCG & Sum‑Squ‑RCG
%  Shared deployment, total power constraint, K = 10
%
%  This script relies on a set of helper functions stored in the same
%  directory.  To run, ensure all files in the `Fig7_RCG` folder are on
%  your MATLAB path.  A CVX installation is required to design the
%  radar‑only 3 dB beampattern (problem (10) in the paper).
%
%  Reference:
%    F. Liu et al., "MU‑MIMO Communications with MIMO Radar:
%    From Co‑existence to Joint Transmission", IEEE Trans. Wireless
%    Communications, vol. 17, no. 4, pp. 2765‑2780, Apr. 2018.
%
%  Author: ChatGPT (translated from pseudo‑code to MATLAB)
% =====================================================================

clear; clc; close all;

%% ------------- System parameters (as in the paper) -----------------
P0_dBm = 20;                  % total BS power in dBm
P0      = 10^(P0_dBm/10);      % linear total power
N       = 20;                  % number of transmit antennas
K       = 10;                  % number of downlink users (fixed)
N0_dBm  = 0;                   % noise power in dBm
N0      = 10^(N0_dBm/10);      % linear noise power
lambda  = 1;                   % wavelength (normalized)
d       = lambda/2;            % half‑wavelength element spacing
MC      = 10;                  % Monte‑Carlo trials (increase for smoother curves)

% Target SINR values in dB for the trade‑off (approx. 5–11 dB in paper)
Gamma_dB_list = 5:1:11;
nGamma        = numel(Gamma_dB_list);

% Weighting vectors (Table II in the paper) for total power constraint
rho_sum_squ = [10, 1];        % [rho1, rho2] for Sum‑Squ penalty
rho_max     = [10, 1];        % [rho1, rho2] for Max penalty

% Epsilon for the log‑sum‑exp approximation used in Max penalty
eps_max_penalty = 0.1;

%% ------------- Steering and radar design parameters ---------------
theta_grid = -90:0.5:90;      % angle grid for beampattern/PSLR evaluation
deg2rad    = pi/180;

% Steering vector function handle a(theta)
steer = @(theta_deg) exp(1j*2*pi*d*sin(theta_deg*deg2rad)*(0:N-1).').';

%% ------------- Step 1: design the radar‑only 3 dB beampattern ----
% Main lobe centered at 0° with 3 dB width from -5° to +5°
theta0 = 0;
theta1 = -5;
theta2 = +5;

% Sidelobe region excludes the main lobe region
theta_sidelobe = [ -90:0.5:-5.5 , 5.5:0.5:90 ];

fprintf('Solving radar‑only 3 dB beampattern (problem (10)) by CVX...\n');
R = solve_radar_3dB_cvx(N, P0, d, theta0, theta1, theta2, theta_sidelobe);

%% ------------- Allocate storage for Monte‑Carlo results ----------
PSLR_sum_squ = zeros(nGamma, MC);
PSLR_max     = zeros(nGamma, MC);

SINR_sum_squ = zeros(nGamma, MC);
SINR_max     = zeros(nGamma, MC);

%% ------------- Monte‑Carlo simulation ---------------------------
fprintf('Running Monte‑Carlo simulation for Fig. 7...\n');

for mc = 1:MC
    % Generate independent Rayleigh flat‑fading channel matrix H (N × K)
    H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);
    
    for ig = 1:nGamma
        Gamma_dB = Gamma_dB_list(ig);
        Gamma_lin = 10^(Gamma_dB/10);    % target SINR in linear scale
        
        % ----- Sum‑Squ RCG (total power constraint) -----
        opts_rcg.maxIter  = 300;
        opts_rcg.tolGrad  = 1e-4;
        opts_rcg.verbose  = 0;
        opts_rcg.stepInit = 1;
        
        T_sum_squ = rcg_sum_squ_total(H, R, Gamma_lin, P0, N0, rho_sum_squ, opts_rcg);
        
        % Compute beampattern and PSLR
        [PSLR_dB, ~] = compute_PSLR(T_sum_squ, theta_grid, steer, theta0, theta1, theta2);
        PSLR_sum_squ(ig, mc) = PSLR_dB;
        
        % Compute average SINR of users (in dB)
        SINR_i = compute_user_SINR(H, T_sum_squ, N0);
        SINR_sum_squ(ig, mc) = 10*log10(mean(SINR_i));
        
        % ----- Max RCG (total power constraint) -----
        T_max = rcg_max_total(H, R, Gamma_lin, P0, N0, rho_max, eps_max_penalty, opts_rcg);
        
        [PSLR_dB2, ~] = compute_PSLR(T_max, theta_grid, steer, theta0, theta1, theta2);
        PSLR_max(ig, mc) = PSLR_dB2;
        
        SINR_i2 = compute_user_SINR(H, T_max, N0);
        SINR_max(ig, mc) = 10*log10(mean(SINR_i2));
    end
end

%% ------------- Average results over Monte‑Carlo trials ----------
avg_PSLR_sum_squ = mean(PSLR_sum_squ, 2);
avg_PSLR_max     = mean(PSLR_max, 2);

avg_SINR_sum_squ = mean(SINR_sum_squ, 2);
avg_SINR_max     = mean(SINR_max, 2);

%% ------------- Plot Fig. 7 (Total power, solid lines) ----------
figure; hold on; grid on; box on;
plot(avg_SINR_max,     avg_PSLR_max,     'r-x', 'LineWidth',1.5, 'DisplayName','Max, RCG (Total)');
plot(avg_SINR_sum_squ, avg_PSLR_sum_squ, 'b-o', 'LineWidth',1.5, 'DisplayName','Sum‑Squ, RCG (Total)');
xlabel('Average SINR (dB)');
ylabel('Average PSLR (dB)');
legend('Location','southwest');
title('Fig. 7 – Trade‑off between PSLR and SINR (RCG, Total Power Constraint)');
