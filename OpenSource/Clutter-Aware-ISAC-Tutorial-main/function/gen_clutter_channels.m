function [H_cold, H_hot, H_uav, uavPara, scatterPara, coldPara, hotPara] = ...
    gen_clutter_channels(para, hotSourcePara, tarPara, cfg)
% Generate cold-clutter, hot-clutter, and UAV channel tensors.
%
% Inputs:
%   para          : Basic system parameter structure
%   hotSourcePara : Hot-source parameter structure
%   tarPara       : Target parameter structure
%   cfg           : Optional configuration structure
%
% Outputs:
%   H_cold        : Cold clutter channel, size Nr x Nt x L x N
%   H_hot         : Hot clutter channel, size Nr x Nt_hs x L x N
%   H_uav         : UAV channel, size Nr x Nt x L x N
%   uavPara       : UAV-related parameter structure
%   scatterPara   : Scatterer parameter structure
%   coldPara      : Cold-clutter parameter structure
%   hotPara       : Hot-clutter parameter structure

if nargin < 4
    cfg = struct;
end

% Basic parameters
N = para.N;
Nt = para.Nt;
Nr = para.Nr;
L = para.L;
Nt_hs = hotSourcePara.Nt_hs;

Pt = para.Pt;
deltaf = para.deltaf;
fc = para.fc;
Tsym = para.Tsym;
c0 = para.c0;
lambda_wave = para.lambda_wave;
PRF = 1 / Tsym;
f_n = para.fc + (0:N-1) * para.deltaf;

% Steering-vector definitions
chi_n = @(n) 1 + n * deltaf / fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);
a_n_hs = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt_hs-1)' * chi_n(n)) ./ sqrt(Nt_hs);

% Default UAV configuration
if ~isfield(cfg, 'uav_num'),            cfg.uav_num = 2; end
if ~isfield(cfg, 'uav_rcs_dB'),         cfg.uav_rcs_dB = [30 40]; end
if ~isfield(cfg, 'uav_thetas'),         cfg.uav_thetas = [-15 30]'; end
if ~isfield(cfg, 'uav_ranges'),         cfg.uav_ranges = [53.2 55.4]'; end
if ~isfield(cfg, 'uav_velos'),          cfg.uav_velos = [61 -31.2]'; end
if ~isfield(cfg, 'uav_ext_enable'),     cfg.uav_ext_enable = false; end

% UAV parameter generation
uavPara.num = cfg.uav_num;
uavPara.rcs = 10.^(cfg.uav_rcs_dB(:) / 10);
uavPara.thetas = cfg.uav_thetas(:);
uavPara.ranges = cfg.uav_ranges(:);
uavPara.velos = cfg.uav_velos(:);
uavPara.delays = 2 * uavPara.ranges / c0;
uavPara.dopplers = 2 * uavPara.velos / lambda_wave;
uavPara.beta_cn = zeros(uavPara.num, N);
for c = 1:uavPara.num
    uavPara.beta_cn(c, :) = sqrt(Pt * (c0 ./ f_n).^2 * para.Gtx * para.Grx ...
        * uavPara.rcs(c) / ((4*pi)^3 * uavPara.ranges(c)^4));
end
uavPara.dopp_nor = uavPara.dopplers / PRF;
uavPara.ext.enable = cfg.uav_ext_enable;

% Default scatterer configuration
if ~isfield(cfg, 'thetac_range'),          cfg.thetac_range = [-90, 90]; end
if ~isfield(cfg, 'scatter_num'),           cfg.scatter_num = 100; end
if ~isfield(cfg, 'scatter_range_inRing'),  cfg.scatter_range_inRing = 35; end
if ~isfield(cfg, 'scatter_range_outRing'), cfg.scatter_range_outRing = 45; end
if ~isfield(cfg, 'scatter_rcs_mean_dB'),   cfg.scatter_rcs_mean_dB = 40; end
if ~isfield(cfg, 'scatter_rcs_std_dB'),    cfg.scatter_rcs_std_dB = 0.5; end

% Scatterer parameter generation
scatterPara.num = cfg.scatter_num;
scatterPara.range_inRing = cfg.scatter_range_inRing;
scatterPara.num_inRing = round(scatterPara.num / 2);
scatterPara.range_outRing = cfg.scatter_range_outRing;
scatterPara.num_outRing = scatterPara.num - scatterPara.num_inRing;
scatterPara.thetas_inRing = scatterAngles(tarPara.theta, ...
    scatterPara.num_inRing, para.avoid_radius, cfg.thetac_range);
scatterPara.thetas_outRing = scatterAngles(tarPara.theta, ...
    scatterPara.num_outRing, para.avoid_radius, cfg.thetac_range);
scatterPara.ranges = [scatterPara.range_inRing * ones(scatterPara.num_inRing, 1); ...
    scatterPara.range_outRing * ones(scatterPara.num_outRing, 1)];
scatterPara.thetas = [scatterPara.thetas_inRing; scatterPara.thetas_outRing];
scatterPara.velos = -1 + 2 * rand(scatterPara.num, 1);
scatterPara.rcs_mean_dB = cfg.scatter_rcs_mean_dB;
scatterPara.rcs_std_dB = cfg.scatter_rcs_std_dB;

% Default cold-clutter configuration
if ~isfield(cfg, 'cold_depression_angle'),    cfg.cold_depression_angle = -5; end
if ~isfield(cfg, 'cold_surface_roughness'),   cfg.cold_surface_roughness = 0.05; end
if ~isfield(cfg, 'cold_terrain_type'),        cfg.cold_terrain_type = 'urban'; end
if ~isfield(cfg, 'cold_coherence_bandwidth'), cfg.cold_coherence_bandwidth = 1e6; end

% Cold-clutter parameter generation
coldPara.depression_angle = cfg.cold_depression_angle;
coldPara.surface_roughness = cfg.cold_surface_roughness;
coldPara.terrain_type = cfg.cold_terrain_type;
coldPara.coherence_bandwidth = cfg.cold_coherence_bandwidth;
coldPara.num = scatterPara.num;
coldPara.thetas = scatterPara.thetas;
coldPara.delays = 2 * scatterPara.ranges / c0;
coldPara.dopplers = 2 * scatterPara.velos / lambda_wave;
coldPara.rcs_dB = scatterPara.rcs_mean_dB + scatterPara.rcs_std_dB * randn(scatterPara.num, 1);
coldPara.rcs = 10.^(coldPara.rcs_dB / 10);
coldPara.beta_cn = zeros(coldPara.num, para.N);
coldPara.sigma2_beta_cn = zeros(coldPara.num, para.N);
coldPara.Rfreq_sqrt = cell(coldPara.num, 1);
for c = 1:coldPara.num
    sigma0_n = clutter_reflectivity_GIT(f_n, coldPara.depression_angle, ...
        coldPara.surface_roughness, coldPara.terrain_type);
    kappa_c_n = para.Pt * para.Gtx * para.Grx * (para.c0 ./ f_n).^2 ...
        * coldPara.rcs(c) / ((4*pi)^3 * scatterPara.ranges(c)^4);
    sigma2_beta = kappa_c_n .* sigma0_n;
    coldPara.sigma2_beta_cn(c, :) = sigma2_beta;

    % Frequency correlation model of clutter coefficients
    sigma2_matrix = sqrt(sigma2_beta(:) * sigma2_beta(:)');
    delta_freq_matrix = abs(bsxfun(@minus, f_n(:), f_n(:)'));
    R_freq = sigma2_matrix .* exp(-delta_freq_matrix / coldPara.coherence_bandwidth);
    coldPara.Rfreq_sqrt{c} = sqrtm(R_freq);
    beta_temp = coldPara.Rfreq_sqrt{c} * (randn(para.N, 1) + 1j * randn(para.N, 1)) / sqrt(2);
    coldPara.beta_cn(c, :) = beta_temp.';
end

% Hot-clutter parameter generation
hotPara.num = hotSourcePara.num * (scatterPara.num + 1);
hotPara.mu_hn = zeros(hotPara.num, N);
hotPara.tx_angles = zeros(hotPara.num, 1);
hotPara.rx_angles = zeros(hotPara.num, 1);
hotPara.delays = zeros(hotPara.num, 1);
hotPara.dopplers = zeros(hotPara.num, 1);
hotPara.rcs_dB = zeros(hotPara.num, 1);
hotPara.rcs = zeros(hotPara.num, 1);
idx = 1;
for hs = 1:hotSourcePara.num
    hs_range = hotSourcePara.ranges(hs);
    hs_theta = deg2rad(hotSourcePara.thetas(hs));
    hs_x = hs_range * cos(hs_theta);
    hs_y = hs_range * sin(hs_theta);
    hs_velo = hotSourcePara.velos(hs);
    hs_vx = hs_velo * cos(hs_theta);
    hs_vy = hs_velo * sin(hs_theta);

    % Direct LOS component from the hot source
    hotPara.tx_angles(idx) = hotSourcePara.thetas(hs);
    hotPara.rx_angles(idx) = hotSourcePara.thetas(hs);
    hotPara.delays(idx) = hs_range / c0;
    hs_speed_proj_direct = hs_vx * cos(hs_theta) + hs_vy * sin(hs_theta);
    hotPara.dopplers(idx) = hs_speed_proj_direct / lambda_wave;
    hotPara.rcs_dB(idx) = 0;
    hotPara.rcs(idx) = 1;
    P_t = hotSourcePara.ILpower(hs);
    hotPara.mu_hn(idx, :) = sqrt(P_t * (c0 ./ f_n).^2 * para.Grx / ((4*pi)^2 * hs_range^2));
    idx = idx + 1;

    % Bistatic hot clutter via environmental scatterers
    for sc = 1:scatterPara.num
        sc_range = scatterPara.ranges(sc);
        sc_theta = deg2rad(scatterPara.thetas(sc));
        sc_x = sc_range * cos(sc_theta);
        sc_y = sc_range * sin(sc_theta);
        sc_velo = scatterPara.velos(sc);
        sc_vx = sc_velo * cos(sc_theta);
        sc_vy = sc_velo * sin(sc_theta);
        dx_tx = sc_x - hs_x;
        dy_tx = sc_y - hs_y;
        tx_angle = atan2(dy_tx, dx_tx);
        rx_angle = sc_theta;

        hotPara.tx_angles(idx) = rad2deg(tx_angle);
        hotPara.rx_angles(idx) = scatterPara.thetas(sc);

        dist_tx = sqrt(dx_tx^2 + dy_tx^2);
        dist_rx = sc_range;
        hotPara.delays(idx) = (dist_tx + dist_rx) / c0;

        hs_speed_proj_tx = hs_vx * cos(tx_angle) + hs_vy * sin(tx_angle);
        sc_speed_proj_tx = sc_vx * cos(tx_angle) + sc_vy * sin(tx_angle);
        sc_speed_proj_rx = sc_vx * cos(rx_angle) + sc_vy * sin(rx_angle);

        f_doppler_tx = (hs_speed_proj_tx - sc_speed_proj_tx) / lambda_wave;
        f_doppler_rx = sc_speed_proj_rx / lambda_wave;
        hotPara.dopplers(idx) = f_doppler_tx + f_doppler_rx;

        hotPara.rcs_dB(idx) = scatterPara.rcs_mean_dB + scatterPara.rcs_std_dB * randn(1);
        hotPara.rcs(idx) = 10^(hotPara.rcs_dB(idx) / 10);
        hotPara.mu_hn(idx, :) = sqrt(P_t * (c0 ./ f_n).^2 * hotPara.rcs(idx) ...
            / ((4*pi)^3 * dist_tx^2 * dist_rx^2));
        idx = idx + 1;
    end
end
hotPara.num_sources = hotSourcePara.num;
hotPara.num_scatters_per_source = scatterPara.num;
hotPara.hot_source_indices = repmat(1:hotSourcePara.num, scatterPara.num, 1);
hotPara.hot_source_indices = hotPara.hot_source_indices(:);
hotPara.scatter_indices = repmat((1:scatterPara.num)', hotSourcePara.num, 1);
%% Generate channel tensors
H_cold = complex(zeros(Nr, Nt, L, N));
H_hot  = complex(zeros(Nr, Nt_hs, L, N));
H_uav  = complex(zeros(Nr, Nt, L, N));
parfor l = 1:L
    Hc_l = complex(zeros(Nr, Nt, N));
    Hh_l = complex(zeros(Nr, Nt_hs, N));
    Hu_l = complex(zeros(Nr, Nt, N));
    for n = 1:N
        % Cold clutter
        temp_cold = 0;
        for c = 1:coldPara.num
            phase_cold = exp(1j * 2*pi * (coldPara.dopplers(c) * (l-1) * Tsym - ...
                (n-1) * deltaf * coldPara.delays(c)));
            temp_cold = temp_cold + coldPara.beta_cn(c, n) * phase_cold ...
                * b_n(coldPara.thetas(c), n-1) * a_n(coldPara.thetas(c), n-1)';
        end
        Hc_l(:, :, n) = temp_cold;

        % Hot clutter
        temp_hot = 0;
        for h = 1:hotPara.num
            phase_hot = exp(1j * 2*pi * (hotPara.dopplers(h) * (l-1) * Tsym - ...
                (n-1) * deltaf * hotPara.delays(h)));
            temp_hot = temp_hot + hotPara.mu_hn(h, n) * phase_hot ...
                * b_n(hotPara.rx_angles(h), n-1) * a_n_hs(hotPara.tx_angles(h), n-1)';
        end
        Hh_l(:, :, n) = temp_hot;

        % UAV return
        temp_uav = 0;
        for u = 1:uavPara.num
            phase_uav = exp(1j * 2*pi * (uavPara.dopplers(u) * (l-1) * Tsym - ...
                (n-1) * deltaf * uavPara.delays(u)));
            temp_uav = temp_uav + uavPara.beta_cn(u, n) * phase_uav ...
                * b_n(uavPara.thetas(u), n-1) * a_n(uavPara.thetas(u), n-1)';
        end
        Hu_l(:, :, n) = temp_uav;
    end
    H_cold(:, :, l, :) = reshape(Hc_l, Nr, Nt, 1, N);
    H_hot(:, :, l, :)  = reshape(Hh_l, Nr, Nt_hs, 1, N);
    H_uav(:, :, l, :)  = reshape(Hu_l, Nr, Nt, 1, N);
end
end