function para = paper_params()
% PAPER_PARAMS  Default simulation parameters used in the demo.
%
% Output:
%   para : Struct containing system dimensions, waveform settings,
%          power/noise levels, algorithm parameters, and plot styles.

% Physical constants
para.c0 = 3e8;                     % Speed of light (m/s)
para.fc = 24e9;                    % Carrier frequency (Hz)
para.lambda = para.c0 / para.fc;   % Wavelength (m)

% MIMO-OFDM dimensions
para.Nt = 6;       % Number of transmit antennas
para.Nr = 6;       % Number of receive antennas
para.Nc = 16;      % Number of subcarriers
para.Ns = 8;       % Number of OFDM symbols
para.K = 2;        % Number of communication users

% Array geometry
para.dT = para.lambda / 2;   % Tx antenna spacing
para.dR = para.lambda / 2;   % Rx antenna spacing

% OFDM parameters
para.bandwidth = 120e6;                 % Total bandwidth (Hz)
para.deltaf = para.bandwidth / para.Nc; % Subcarrier spacing (Hz)
para.T = 1 / para.deltaf;               % Useful OFDM symbol duration
para.Tcp = para.T/4;                    % Cyclic prefix duration
para.Tsym = para.T + para.Tcp;          % Total OFDM symbol duration

% Modulation / CI parameters
para.Mpsk = 4;                  % M-PSK order
para.phi = pi / para.Mpsk;      % Decision-region half-angle

% Communication QoS threshold
para.Gamma_dB = 6;
para.Gamma_lin = 10^(para.Gamma_dB / 10);

% Noise power
para.sigma_c2 = 10.^((-70 - 30) / 10);   % Communication noise power
para.sigma_r2 = 10.^((-70 - 30) / 10);   % Radar noise power
para.sigma_c = sqrt(para.sigma_c2);
para.sigma_r = sqrt(para.sigma_r2);

% Transmit power / illumination settings
para.PT = 10*para.Ns;          % Total transmit power
para.P0 = 8*para.Ns;           % Desired target illumination parameter
para.delta_th = 1e-5;          % Generic stopping threshold
para.wc = 1;                   % Combined waveform weight

% Derived dimensions and constants
para.Ntot = para.Ns * para.Nc * para.Nt;      % Total waveform length
para.M = para.Ns * para.Nc;                   % Total delay-Doppler bins
para.amp = sqrt(para.PT / para.Ntot);         % Constant-modulus amplitude
para.Pbar0 = para.Nt * para.PT - para.P0;     % Reformulated illumination threshold
para.lambdaC = para.Nt^2;                     % Largest eigenvalue bound in MM

% Target direction
para.theta0_deg = 10;
para.theta0_rad = para.theta0_deg * pi / 180;

% Large-scale fading model
para.pathloss_zeta0_dB = -30;
para.pathloss_zeta0 = 10^(para.pathloss_zeta0_dB / 10);
para.d0 = 1;
para.epsilon = 2.6;
para.user_dist_min = 30;
para.user_dist_max = 100;

% CVX settings
para.use_cvx = true;
para.cvx_quiet = true;
para.cvx_solver = 'Mosek_5';
para.show_progress = true;

% MM-ADMM parameters
para.rho = 1;                   % Initial ADMM penalty parameter
para.outer_max = 2e3;           % Maximum MM outer iterations
para.inner_max = 300;           % Maximum ADMM inner iterations
para.squarem_v_tol = 1e-14;     % Reserved tolerance for acceleration
para.outer_max_radar = 50;

% ALM-RCG parameters
para.alm_rho0 = 20;             % Initial ALM penalty parameter
para.alm_rho_max = 50;          % Maximum ALM penalty parameter
para.alm_outer_max = 200;       % Maximum ALM outer iterations
para.show_progress_alm = 1;
para.rcg_maxiter = 200;         % Maximum RCG inner iterations
para.rcg_tolgradnorm = 1e-4;    % RCG gradient tolerance
para.rcg_minstepsize = 1e-4;    % Minimum line-search step size
para.rcg_verbosity = 0;

% Plot styles
para.markers = {'o', 's', 'p', '^', 'd', 'x'};
para.colors = {'#0072BD', '#77AC30',  '#9999FF', '#D95319','#EDB120', '#E67DAF'};
end
