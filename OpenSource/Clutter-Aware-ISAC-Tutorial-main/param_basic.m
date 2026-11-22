% This Matlab script is used to generate the basic system parameters for the
% clutter-aware ISAC simulation framework in the paper:
% R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-aware integrated sensing and communication: Models, methods, and future directions,” Proc. IEEE, to appear.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-18

clear; close all; clc;
rng('shuffle');
%% Basic physical constants and OFDM-related parameters
fc = 28e9;                     % Carrier frequency (Hz)
c0 = 3e8;                      % Speed of light (m/s)
Boltz = 1.380649e-23;          % Boltzmann constant (J/K)
sc_confi = 3;                  % Subcarrier spacing configuration index
kaapa = 64;                    % Scaling factor used in cyclic prefix calculation
deltaf = 2^(sc_confi) * 15e3;  % Subcarrier spacing (Hz)
Nt = 16;                       % Number of transmit antennas
Nr = 16;                       % Number of receive antennas
N = 512;                       % Number of subcarriers
L = 256;                       % Number of OFDM symbols / slow-time snapshots
T = 1 / deltaf;                                % Useful symbol duration (s)
Tchip = 1 / (480e3 * 4096);                    % Basic chip duration (s)
Tcp = 144 * kaapa * 2^(-sc_confi) * Tchip;     % Cyclic prefix duration (s)
Tsym = T + Tcp;                                % Total OFDM symbol duration (s)
F = 10^(3/10);                 % Receiver noise figure (linear scale)
T_temp = 290;                  % System noise temperature (K)
QAM_order = 64;                % QAM modulation order
sigma2 = F * Boltz * deltaf * T_temp;  % Noise power per subcarrier
Pt = 10^(53/10) / 1e3;         % Transmit power (W), converted from 53 dBm
Ns = Nt;                       % Number of transmitted streams
Ku = 3;                        % Number of communication users
lambda_wave = c0 / fc;         % Wavelength (m)
bandwidth = N * deltaf;        % Total signal bandwidth (Hz)
range_res = c0 / bandwidth / 2;        % Range resolution (m)
velo_res = lambda_wave / L / Tsym / 2; % Velocity resolution (m/s)
angle_res = 101.6 / Nr;                % Approximate angle resolution (deg)
avoid_radius = 0.5;            % Safety/avoidance radius (deg)
Gtx = 10^(10/10);              % Transmit antenna gain (linear scale)
Grx = 10^(10/10);              % Receive antenna gain (linear scale)
para = struct('fc', fc, 'c0', c0, 'Boltz', Boltz, 'sc_confi', sc_confi, 'kaapa', kaapa, ...
    'deltaf', deltaf, 'N', N, 'L', L, 'T', T, 'F', F, 'T_temp', T_temp, 'angle_res', angle_res,...
    'QAM_order', QAM_order, 'Tchip', Tchip, 'lambda_wave', lambda_wave, ...
    'bandwidth', bandwidth, 'Tcp', Tcp, 'Tsym', Tsym, 'sigma2', sigma2,...
    'Nt', Nt, 'Nr', Nr, 'Pt', Pt, 'Ns', Ns, 'Ku', Ku, 'Gtx', Gtx, 'Grx', Grx, ...
    'range_res', range_res, 'velo_res', velo_res, 'avoid_radius', avoid_radius);
f_n = para.fc + (0:para.N-1) * para.deltaf;   % Frequency of each subcarrier
%% Target parameter settings
tarPara.M = 1;                                % Number of sensing targets
tarPara.rcs = 10.^([-13] / 10);               % Target RCS (linear scale)
tarPara.theta = [-10]';                       % Target angle (deg)
tarPara.ranges = [41.8]';                     % Target range (m)
tarPara.velos = [-31.2]';                     % Target radial velocity (m/s)
tarPara.delays = 2 * tarPara.ranges / para.c0;              % Round-trip delay (s)
tarPara.dopplers = 2 * tarPara.velos / para.lambda_wave;    % Doppler frequency (Hz)
tarPara.alpha_mn = zeros(tarPara.M, para.N);
for m = 1:tarPara.M
    tarPara.alpha_mn(m, :) = sqrt(para.Pt * (para.c0./f_n).^2*para.Gtx*para.Grx ...
        * tarPara.rcs(m) / ((4*pi)^3 * tarPara.ranges(m)^4));
end

%% Hot source interference settings
hotSourcePara.num = 1;                         % Number of hot interference sources
hotSourcePara.Nt_hs = 16;                      % Number of antennas at hot source
hotSourcePara.thetas = [-74.5];                % Hot source angle(s) (deg)
hotSourcePara.ranges = [122.4];                % Hot source range(s) (m)
hotSourcePara.velos = [0];                     % Hot source velocity(ies) (m/s)
hotSourcePara.ILpower = 10.^([53] / 10) / 1e3; % Interference leakage power (W)

%% Extended scenario configuration: UAVs and clutter region
cfg.uav_num = 2;                              % Number of UAV-like extended targets
cfg.uav_rcs_dB = [30 40];                     % UAV RCS values (dBsm)
cfg.uav_thetas = [-15 30]';                   % UAV angles (deg)
cfg.uav_ranges = [53.2 55.4]';                % UAV ranges (m)
cfg.uav_velos = [61 -31.2]';                  % UAV velocities (m/s)
cfg.uav_ext_enable = false;                   % Enable/disable extended UAV model

cfg.thetac_range = [-90, 90];                 % Angular range of clutter region (deg)
cfg.scatter_num = 100;                        % Number of discrete clutter scatterers
cfg.scatter_range_inRing = 35;                % Inner radius of clutter ring (m)
cfg.scatter_range_outRing = 45;               % Outer radius of clutter ring (m)
cfg.scatter_rcs_mean_dB = 40;                 % Mean scatterer RCS (dB)
cfg.scatter_rcs_std_dB = 0.5;                 % Std of scatterer RCS (dB)

cfg.cold_depression_angle = -5;               % Depression angle for ground clutter (deg)
cfg.cold_surface_roughness = 0.05;            % Surface roughness coefficient
cfg.cold_terrain_type = 'urban';              % Terrain type for clutter modeling
cfg.cold_coherence_bandwidth = 1e6;           % Coherence bandwidth of cold clutter (Hz)
%%
filePath = fullfile('data', 'parameters_basic.mat');
[fileDir, ~, ~] = fileparts(filePath);
mkdir(fileDir);
save(filePath, 'para', 'tarPara', 'hotSourcePara', 'cfg');