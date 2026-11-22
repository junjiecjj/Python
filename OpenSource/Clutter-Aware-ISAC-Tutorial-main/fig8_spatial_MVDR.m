% This Matlab script can be used to generate Fig. 8 in the paper:
% R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-aware integrated sensing and communication: Models, methods, and future directions,” Proc. IEEE, to appear.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-18

clear;
clc;
close all;
rng('shuffle');

addpath('function');
colors = {'#0072BD', '#77AC30', '#7E2F8E', '#D95319', '#E67DAF', '#EDB120', '#A21E2D'};
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'function'));
load(fullfile(root,'data','parameters_basic.mat'), 'para', 'tarPara', 'hotSourcePara', 'cfg')
%% Basic system parameters
Nt = para.Nt;                  % Number of transmit antennas
Nr = para.Nr;                  % Number of receive antennas
N = para.N;                    % Number of subcarriers
L = para.L;                    % Number of OFDM symbols / slow-time snapshots
Ntr = 8 * Nr;                  % Number of sufficient training snapshots
Ntr_insuff = 2 * Nr;           % Number of insufficient training snapshots
para.Ntr = Ntr;
para.Ntr_insuff = Ntr_insuff;
Ns = para.Ns;                  % Number of transmit streams
Pt = para.Pt;                  % Total transmit power
deltaf = para.deltaf;          % Subcarrier spacing
fc = para.fc;                  % Carrier frequency
Tsym = para.Tsym;              % OFDM symbol duration
Nt_hs = hotSourcePara.Nt_hs;   % Number of hot-source antennas
f_n = para.fc + (0:para.N-1) * para.deltaf;   % Frequency grid
para.AoAerror = -1;            % Assumed target AoA mismatch (deg)
PRF = 1 / para.Tsym;           % Pulse repetition frequency

% Frequency-dependent steering vectors
chi_n = @(n) 1 + n * deltaf / fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);

% Generate clutter channels
[H_cold, H_hot, ~, uavPara] = gen_clutter_channels(para, hotSourcePara, tarPara, cfg);

% Generate extended UAV channel
uavPara.ext.enable = true;
uavPara.ext.theta_span_deg = [para.angle_res / 3; para.angle_res / 3];
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
%% Generate transmit signals
W_total = zeros(Nt, Ns, N);
for n = 1:N
    Wn = (randn(Nt, Ns) + 1j * randn(Nt, Ns)) / sqrt(2);
    W_total(:, :, n) = Wn * sqrt(Pt / N / norm(Wn, 'fro')^2);
end
bits_sens = randi([0, 3], Ns, N);
s_total = qammod(bits_sens, 4, 'UnitAveragePower', true);
s_total = repmat(s_total, [1, 1, L]);
x_nl = zeros(Nt, N, L);
for n = 1:N
    x_nl(:, n, :) = W_total(:, :, n) * squeeze(s_total(:, n, :));
end
x_hs = sqrt(hotSourcePara.ILpower(1) / N / Nt_hs / 2) * ...
    (randn(Nt_hs, N) + 1j * randn(Nt_hs, N));
x_hs = repmat(x_hs, [1, 1, L]);
%% Generate echoes and interference
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
%% Generate target echo with AoA mismatch
y_tar_error = zeros(Nr, N, L);
for i = 1:Nr
    temp = 0;
    for m = 1:tarPara.M
        A_tx = a_n(tarPara.theta(m) + para.AoAerror, 0:N-1);
        A_tx_expanded = repmat(A_tx, [1, 1, L]);
        s_nl = squeeze(sum(conj(A_tx_expanded) .* x_nl, 1));
        B_rx = b_n(tarPara.theta(m) + para.AoAerror, 0:N-1);
        delay_vec = exp(-1j * 2 * pi * f_n' * tarPara.delays(m));
        dopp_vec = exp(-1j * 2 * pi * (tarPara.dopplers(m) * (0:L-1)' * para.Tsym));
        temp = temp + (tarPara.alpha_mn(m, :).' .* B_rx(i, :).' .* delay_vec) ...
            * dopp_vec' .* s_nl;
    end
    y_tar_error(i, :, :) = temp;
end
%% Estimate covariance matrices
y_clut = y_uav + y_cold + y_hot + noise;            % Clutter-only snapshots
y_echo = y_tar + y_uav + y_cold + y_hot + noise;    % Total received signal
y_echo_diff = y_echo - y_tar_error;                 % Signal with target mismatch removed

R_ideal = zeros(Nr, Nr, N);
R_insufficient = zeros(Nr, Nr, N);
R_withTar_AoAerr = zeros(Nr, Nr, N);
for n = 1:N
    % Ideal clutter covariance with sufficient training snapshots
    y_clut_train = squeeze(y_clut(:, n, 1:Ntr));
    R_ideal(:, :, n) = (y_clut_train * y_clut_train') / Ntr;

    % Clutter covariance with insufficient training snapshots
    R_insufficient(:, :, n) = (y_clut_train(:, 1:Ntr_insuff) * y_clut_train(:, 1:Ntr_insuff)') / Ntr_insuff;
    
    % Covariance estimated under target AoA mismatch
    y_echo_diff_train = squeeze(y_echo_diff(:, n, 1:Ntr));
    R_withTar_AoAerr(:, :, n) = (y_echo_diff_train * y_echo_diff_train') / Ntr;
end
%% Design MVDR receive beamformers
w_mvdr = zeros(Nr, N);
w_mvdr_insuff = zeros(Nr, N);
w_mvdr_withTarAoA = zeros(Nr, N);
for n = 1:N
    b_tar = b_n(tarPara.theta, n-1);

    % MVDR with ideal covariance
    w_mvdr(:, n) = (R_ideal(:, :, n) \ b_tar) / (b_tar' * (R_ideal(:, :, n) \ b_tar));

    % MVDR with insufficient training snapshots
    w_mvdr_insuff(:, n) = (R_insufficient(:, :, n) \ b_tar) / ...
        (b_tar' * (R_insufficient(:, :, n) \ b_tar));

    % MVDR with target AoA mismatch
    w_mvdr_withTarAoA(:, n) = (R_withTar_AoAerr(:, :, n) \ b_tar) / ...
        (b_tar' * (R_withTar_AoAerr(:, :, n) \ b_tar));
end
w_mvdr = w_mvdr * sqrt(N * Nr / norm(w_mvdr, 'fro')^2);
w_mvdr_insuff = w_mvdr_insuff * sqrt(N * Nr / norm(w_mvdr_insuff, 'fro')^2);
w_mvdr_withTarAoA = w_mvdr_withTarAoA * sqrt(N * Nr / norm(w_mvdr_withTarAoA, 'fro')^2);
%% Compute receive beampatterns
theta_range = -90:1:90;
beampatt_mvdr = zeros(size(theta_range));
beampatt_mvdr_insuff = zeros(size(theta_range));
beampatt_mvdr_withTarAoA = zeros(size(theta_range));
for ii = 1:length(theta_range)
    Btheta = b_n(theta_range(ii), 0:N-1);
    beampatt_mvdr(ii) = sum(abs(sum(conj(w_mvdr) .* Btheta, 1)).^2);
    beampatt_mvdr_insuff(ii) = sum(abs(sum(conj(w_mvdr_insuff) .* Btheta, 1)).^2);
    beampatt_mvdr_withTarAoA(ii) = sum(abs(sum(conj(w_mvdr_withTarAoA) .* Btheta, 1)).^2);
end
beampatt_mvdr_dB = 10 * log10(beampatt_mvdr / max(beampatt_mvdr));
beampatt_imper_dB = 10 * log10(beampatt_mvdr_insuff / max(beampatt_mvdr_insuff));
beampatt_mvdr_withTarAoA_dB = 10 * log10(beampatt_mvdr_withTarAoA / max(beampatt_mvdr_withTarAoA));
%% Plot receive beampatterns
h_rxbeam = figure('Color', 'white');
pt = plot([tarPara.theta, tarPara.theta], [-100, 0], 'g--', 'LineWidth', 1.5);
hold on;
p_ideal = plot(theta_range, beampatt_mvdr_dB, 'LineWidth', 1.5, 'Color', colors{1});
p_insuff = plot(theta_range, beampatt_imper_dB, 'LineWidth', 1.5, 'Color', colors{2}, 'LineStyle', '-');
p_withAoA = plot(theta_range, beampatt_mvdr_withTarAoA_dB, 'LineWidth', 1.5, 'Color', colors{4}, 'LineStyle', '-');
plot([uavPara.thetas(1), uavPara.thetas(1)], [-150, 10], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
plot([uavPara.thetas(2), uavPara.thetas(2)], [-150, 10], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
plot([-74.5, -74.5], [-150, 10], 'Color', 'm', 'LineStyle', '--', 'LineWidth', 1.5);
xlabel('Angle (°)', 'FontSize', 12);
ylabel('Beampattern (dB)', 'FontSize', 12);
grid on;
xlim([-90, 90]);
ylim([-30, 2]);
legend([p_ideal, p_insuff, p_withAoA], ...
    {'MVDR ($N_{\rm{tr}} = 128$)', 'MVDR ($N_{\rm{tr}} = 32$)', 'MVDR (AoA mismatch)'}, ...
    'Interpreter', 'latex', 'FontSize', 11, ...
    'Position', [0.130634475171813 0.108412702840479 0.374220103953391 0.137619043191275]);

axes('Position', [0.585141093474418 0.821428571428572 0.152358906525582 0.0902337309669428]);
plot(theta_range, beampatt_mvdr_dB, 'LineWidth', 1.5, 'Color', colors{1});
hold on;
plot(theta_range, beampatt_imper_dB, 'LineWidth', 1.5, 'Color', colors{2}, 'LineStyle', '-');
plot(theta_range, beampatt_mvdr_withTarAoA_dB, 'LineWidth', 1.5, 'Color', colors{4}, 'LineStyle', '-');
annotation('arrow', [0.507142857142857 0.558928571428571], ...
    [0.872809523809524 0.871428571428572], 'HeadWidth', 8, 'HeadLength', 8);
annotation('rectangle', ...
    [0.453571428571429 0.835714285714286 0.05 0.0880952380952412], 'LineWidth', 1);
hold off;
box on;
grid on;
xlim([-15 -5]);
ylim([-5 2]);