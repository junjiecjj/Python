% This Matlab script can be used to generate Fig. 10 in the paper:
% R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-aware integrated sensing and communication: Models, methods, and future directions,” Proc. IEEE, to appear.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-18

clear;
clc;
close all;
rng('shuffle');

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'function'));
load(fullfile(root,'data','parameters_basic.mat'), 'para', 'hotSourcePara', 'cfg')
%% Basic system parameters
N = para.N;          % Number of subcarriers
Nt = 8;              % Number of transmit antennas
Nr = 16;             % Number of receive antennas
L = 128;             % Number of OFDM symbols
Nt_hs = 8;           % Number of hot-source antennas
para.Nt = Nt;
para.Nr = Nr;
para.L = L;
hotSourcePara.Nt_hs = Nt_hs;
Ku = para.Ku;                  % Number of communication users
Ns = Nt;                       % Number of transmit streams
Pt = para.Pt;                  % Total transmit power
deltaf = para.deltaf;          % Subcarrier spacing
fc = para.fc;                  % Carrier frequency
Tsym = para.Tsym;              % OFDM symbol duration
QAM_order = para.QAM_order;    % QAM order for communication symbols
PRF = 1 / Tsym;                % Pulse repetition frequency
n0 = 1;                        % Reference subcarrier index
f_n = para.fc + (0:para.N-1) * para.deltaf;   % Frequency grid

% Frequency-dependent steering vectors
chi_n = @(n) 1 + n * deltaf / fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);
a_n_hs = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt_hs-1)' * chi_n(n)) ./ sqrt(Nt_hs);

% Target parameters
tarPara.M = 1;
tarPara.rcs = 10.^([3] / 10);
tarPara.theta = [-10]';
tarPara.ranges = [41.8]';
tarPara.delays = 2 * tarPara.ranges / para.c0;
tarPara.velos = [-31.2]';
tarPara.dopplers = 2 * tarPara.velos / para.lambda_wave;
tarPara.alpha_mn = zeros(tarPara.M, para.N);
for m = 1:tarPara.M
    tarPara.alpha_mn(m, :) = sqrt(para.Pt * (para.Gtx * para.Grx) * (para.c0 ./ f_n).^2 ...
        * tarPara.rcs(m) / ((4*pi)^3 * tarPara.ranges(m)^4));
end
tarPara.dopp_nor = tarPara.dopplers / PRF;

% Generate clutter channels
[H_cold, H_hot, ~, uavPara, ~, coldPara, hotPara] = ...
    gen_clutter_channels(para, hotSourcePara, tarPara, cfg);
%% Generate extended UAV channel model
uavPara.ext.enable = true;
uavPara.ext.theta_span_deg = [para.angle_res / 2; para.angle_res / 2];
uavPara.ext.dopp_span_nor = [1 / L / 2; 1 / L / 2];
uavPara.ext.Kth = 7;
uavPara.ext.Knu = 5;
uavPara.ext.theta_list = cell(uavPara.num, 1);
uavPara.ext.dopp_nor_list = cell(uavPara.num, 1);
uavPara.ext.w_list = cell(uavPara.num, 1);
for u = 1:uavPara.num
    dth = linspace(-uavPara.ext.theta_span_deg(u), uavPara.ext.theta_span_deg(u), uavPara.ext.Kth);
    dnu = linspace(-uavPara.ext.dopp_span_nor(u), uavPara.ext.dopp_span_nor(u), uavPara.ext.Knu);
    [TH, NU] = ndgrid(uavPara.thetas(u) + dth, uavPara.dopp_nor(u) + dnu);
    uavPara.ext.theta_list{u} = TH(:);
    uavPara.ext.dopp_nor_list{u} = NU(:);
    K = numel(TH);
    uavPara.ext.w_list{u} = ones(K, 1) / K;
end
H_uav = zeros(Nr, Nt, L, N);
parfor l = 1:L
    for n = 1:N
        temp_uav = zeros(Nr, Nt);
        for u = 1:uavPara.num
            delay_u = uavPara.delays(u);
            beta0_n = uavPara.beta_cn(u, n);
            th_list = uavPara.ext.theta_list{u};
            nu_list = uavPara.ext.dopp_nor_list{u};
            w_list = uavPara.ext.w_list{u};
            for k = 1:length(th_list)
                theta_k = th_list(k);
                fd_k = nu_list(k) * PRF;
                wk = w_list(k);
                beta_k_n = beta0_n * sqrt(wk);
                phase_k = exp(1j * 2 * pi * (fd_k * (l-1) * Tsym - (n-1) * deltaf * delay_u));
                temp_uav = temp_uav + beta_k_n * phase_k * b_n(theta_k, n-1) * a_n(theta_k, n-1)';
            end
        end
        H_uav(:, :, l, n) = temp_uav;
    end
end
%% Generate transmit signals for the reference subcarrier
Hcomm_n0 = (randn(Nt, Ku) + 1j * randn(Nt, Ku)) / sqrt(2);
W_total_n0 = zeros(Nt, Ns);
for u = 1:Ku
    W_total_n0(:, u) = Hcomm_n0(:, u) / norm(Hcomm_n0(:, u));
end
for s = 1:Ns-Ku
    target_idx = mod(s-1, tarPara.M) + 1;
    W_total_n0(:, Ku+s) = a_n(tarPara.theta(target_idx), n0-1);
end
W_total_n0 = W_total_n0 * sqrt(Pt / N / norm(W_total_n0, 'fro')^2);

bits_comm = randi([0, QAM_order-1], Ku, L);
s_c_n0 = qammod(bits_comm, QAM_order, 'UnitAveragePower', true);
bits_sensing = randi([0, 3], Ns-Ku, L);
s_s_n0 = qammod(bits_sensing, 4, 'UnitAveragePower', true);
s_total_n0 = [s_c_n0; s_s_n0];
x_n0l = W_total_n0 * s_total_n0;
x_hs_n0 = sqrt(hotSourcePara.ILpower / Nt_hs / N / 2) * ...
    (randn(Nt_hs, L) + 1j * randn(Nt_hs, L));

% Build block matrices for STAP processing
X_blocks = cell(L, 1);
H_blocks = cell(L, 1);
for l = 1:L
    X_blocks{l} = kron(eye(Nr), x_n0l(:, l).');
    H_blocks{l} = kron(eye(Nr), x_hs_n0(:, l).');
end
X_n0 = blkdiag(X_blocks{:});
H_n0 = blkdiag(H_blocks{:});
%% Compute clutter-plus-interference covariance matrix
Rcc_n0 = zeros(Nr * L, Nr * L);
Rhc_n0 = zeros(Nr * L, Nr * L);
tvec = (0:L-1).' * Tsym;

% Cold clutter
sigma2_cold = abs(coldPara.beta_cn).^2;
for c = 1:coldPara.num
    theta_c = coldPara.thetas(c);
    fd_c = coldPara.dopplers(c);
    sigma2 = sigma2_cold(c, n0);
    a_c = a_n(theta_c, n0-1);
    b_c = b_n(theta_c, n0-1);
    d_c = exp(1j * 2 * pi * fd_c * tvec);
    z_c = a_c' * x_n0l;
    S_c = b_c * z_c;
    g_c = reshape(S_c .* d_c.', Nr * L, 1);
    Rcc_n0 = Rcc_n0 + sigma2 * (g_c * g_c');
end

% UAV clutter
sigma2_uav = abs(uavPara.beta_cn).^2;
for u = 1:uavPara.num
    sigma2_total = sigma2_uav(u, n0);
    th_list = uavPara.ext.theta_list{u};
    nu_list = uavPara.ext.dopp_nor_list{u};
    w_list = uavPara.ext.w_list{u};
    fd_list = nu_list * PRF;
    for k = 1:length(th_list)
        theta_k = th_list(k);
        sigma2_k = sigma2_total * w_list(k);
        fd_k = fd_list(k);
        a_k = a_n(theta_k, n0-1);
        b_k = b_n(theta_k, n0-1);
        d_k = exp(1j * 2 * pi * fd_k * tvec);
        z_k = a_k' * x_n0l;
        S_k = b_k * z_k;
        g_k = reshape(S_k .* d_k.', Nr * L, 1);
        Rcc_n0 = Rcc_n0 + sigma2_k * (g_k * g_k');
    end
end

% Hot clutter
sigma2_hot = abs(hotPara.mu_hn).^2;
for h = 1:hotPara.num
    theta_t = hotPara.tx_angles(h);
    theta_r = hotPara.rx_angles(h);
    fd_h = hotPara.dopplers(h);
    sigma2 = sigma2_hot(h, n0);
    a_h = a_n_hs(theta_t, n0-1);
    b_h = b_n(theta_r, n0-1);
    d_h = exp(1j * 2 * pi * fd_h * tvec);
    z_h = a_h' * x_hs_n0;
    S_h = b_h * z_h;
    g_h = reshape(S_h .* d_h.', Nr * L, 1);
    Rhc_n0 = Rhc_n0 + sigma2 * (g_h * g_h');
end
RI_n0 = Rcc_n0 + Rhc_n0 + para.sigma2 * eye(Nr * L);
%% Generate the received signal tensor
x_nl = repmat(reshape(x_n0l, Nt, 1, []), [1, N, 1]);
x_hs = repmat(reshape(x_hs_n0, Nt_hs, 1, []), [1, N, 1]);

y_tar = echo_tar(para, tarPara, x_nl, a_n, b_n);
y_uav = zeros(Nr, N, L);
y_cold = zeros(Nr, N, L);
y_hot = zeros(Nr, N, L);
for l = 1:L
    for n = 1:N
        y_uav(:, n, l) = H_uav(:, :, l, n) * x_nl(:, n, l);
        y_cold(:, n, l) = H_cold(:, :, l, n) * x_nl(:, n, l);
        y_hot(:, n, l) = H_hot(:, :, l, n) * x_hs(:, n, l);
    end
end
noise = sqrt(para.sigma2 / 2) * (randn(Nr, N, L) + 1j * randn(Nr, N, L));
y_echo = y_tar + y_uav + y_cold + y_hot + noise;
%% STAP preprocessing
R_clut_inv = inv(RI_n0);
[U_all, D_all] = eig(RI_n0, 'vector');
[D_sorted, idx] = sort(real(D_all), 'descend');
U_sorted = U_all(:, idx);
r_rr = 500;
U_r = U_sorted(:, 1:r_rr);
Lambda_r = D_sorted(1:r_rr);
y_n0 = reshape(y_echo(:, n0, :), [], 1);

% Compute angle-Doppler maps
theta_scan = linspace(-90, 90, 181);
doppler_scan_nor = linspace(-0.5, 0.5, L+1);
doppler_scan = doppler_scan_nor * PRF;

ADM_normSTAP = zeros(length(theta_scan), length(doppler_scan));
ADM_STAP = zeros(length(theta_scan), length(doppler_scan));
ADM_RRSTAP = zeros(length(theta_scan), length(doppler_scan));
for ii = 1:length(theta_scan)
    a_theta = a_n(theta_scan(ii), n0-1);
    b_theta = b_n(theta_scan(ii), n0-1);
    for jj = 1:length(doppler_scan)
        d_dopp = exp(1j * 2 * pi * doppler_scan(jj) * (0:L-1)' * para.Tsym);
        v_n0X = X_n0 * kron(d_dopp, kron(b_theta, conj(a_theta)));
        % Conventional processing
        w_normSTAP = v_n0X;
        w_normSTAP = w_normSTAP * sqrt(Nr * L / norm(w_normSTAP, 'fro')^2);

        % Full-rank STAP
        w_STAP = (R_clut_inv * v_n0X) / (v_n0X' * R_clut_inv * v_n0X);
        w_STAP = w_STAP * sqrt(Nr * L / norm(w_STAP, 'fro')^2);

        % Reduced-rank STAP
        proj = U_r' * v_n0X;
        tmp = U_r * ((1 ./ Lambda_r) .* proj);
        w_RRSTAP = tmp / (v_n0X' * tmp);
        w_RRSTAP = w_RRSTAP * sqrt(Nr * L / norm(w_RRSTAP, 'fro')^2);
        
        ADM_normSTAP(ii, jj) = abs(w_normSTAP' * y_n0)^2;
        ADM_STAP(ii, jj) = abs(w_STAP' * y_n0)^2;
        ADM_RRSTAP(ii, jj) = abs(w_RRSTAP' * y_n0)^2;
    end
end

% Normalize the ADMs
zmax = max([ADM_normSTAP, ADM_STAP, ADM_RRSTAP], [], 'all');
ADM_normSTAP_dB = 10 * log10(ADM_normSTAP / zmax);
ADM_STAP_dB = 10 * log10(ADM_STAP / zmax);
ADM_RRSTAP_dB = 10 * log10(ADM_RRSTAP / zmax);
%% Plot ADM without STAP
theta_h = 2 * 101.6 / para.Nr;
dopp_w = 2 / para.L;
% zmin = min([ADM_normSTAP_dB, ADM_STAP_dB, ADM_RRSTAP_dB], [], 'all');
zmin = -90;
h_mf = figure('Color', 'white');
imagesc(doppler_scan_nor, theta_scan, ADM_normSTAP_dB);
hold on;
xlim([-0.2, 0.2]);
ylim([-90, 90]);
zlim([zmin, 0]);
clim([zmin 0]);
colormap('jet');
shading interp;
colorbar;
xlabel('Normalized Doppler', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Angle (°)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
annotRectData(gca, tarPara.dopp_nor, tarPara.theta, dopp_w, theta_h, 'Color', 'k', 'LineWidth', 1.3);
annotEllipseData(gca, uavPara.dopp_nor(1), uavPara.thetas(1), dopp_w, theta_h, 'Color', 'k', 'LineWidth', 1.25);
annotEllipseData(gca, uavPara.dopp_nor(2), uavPara.thetas(2), dopp_w, theta_h, 'Color', 'k', 'LineWidth', 1.25);

% Plot ADM with full-rank STAP
h_stap = figure('Color', 'white');
imagesc(doppler_scan_nor, theta_scan, ADM_STAP_dB);
hold on;
xlim([-0.2, 0.2]);
ylim([-90, 90]);
zlim([zmin, 0]);
clim([zmin 0]);
colormap('jet');
shading interp;
colorbar;
xlabel('Normalized Doppler', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Angle (°)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
annotRectData(gca, tarPara.dopp_nor, tarPara.theta, dopp_w, theta_h, 'Color', 'r', 'LineWidth', 1.3);
annotEllipseData(gca, uavPara.dopp_nor(1), uavPara.thetas(1), dopp_w, theta_h, 'Color', 'r', 'LineWidth', 1.25);
annotEllipseData(gca, uavPara.dopp_nor(2), uavPara.thetas(2), dopp_w, theta_h, 'Color', 'r', 'LineWidth', 1.25);

% Plot ADM with reduced-rank STAP
h_rrstap = figure('Color', 'white');
imagesc(doppler_scan_nor, theta_scan, ADM_RRSTAP_dB);
hold on;
xlim([-0.2, 0.2]);
ylim([-90, 90]);
zlim([zmin, 0]);
clim([zmin 0]);
colormap('jet');
shading interp;
colorbar;
xlabel('Normalized Doppler', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Angle (°)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
annotRectData(gca, tarPara.dopp_nor, tarPara.theta, dopp_w, theta_h, 'Color', 'r', 'LineWidth', 1.3);
annotEllipseData(gca, uavPara.dopp_nor(1), uavPara.thetas(1), dopp_w, theta_h, 'Color', 'r', 'LineWidth', 1.25);
annotEllipseData(gca, uavPara.dopp_nor(2), uavPara.thetas(2), dopp_w, theta_h, 'Color', 'r', 'LineWidth', 1.25);