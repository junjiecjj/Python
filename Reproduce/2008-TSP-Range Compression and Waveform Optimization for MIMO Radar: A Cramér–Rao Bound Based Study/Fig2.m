

clc;
clear all;
close all;


% init_params.m - 通用参数
% init_params.m
% 所有脚本共用参数

N = 10;         % 发射天线数
M = 10;         % 接收天线数
L = 256;        % 快拍数（波形长度）
P = 1;          % 总发射功率
ASNR_dB = 40;   % 阵列信噪比 (dB)
sigma2 = (M*N*P) / (10^(ASNR_dB/10));   % 噪声方差

% 干扰机参数
jammer_angle = -5;          % 度
AINR_dB = 100;              % 阵列干扰噪比 (dB)
jammer_power = 10^(AINR_dB/10) * sigma2 / M;

% 目标参数（单目标情况）
theta_t = -16.5;            % 目标角度 (deg)
b_t = 1;                    % 目标复振幅（实数）

% 生成正交波形矩阵 Phi (N x L)
% 使用 QR 分解生成随机正交矩阵，满足功率约束
[Phi, ~] = qr(randn(N, L) + 1j*randn(N, L), 0);
Phi = sqrt(P/N) * Phi;      % 每行功率 = P/N

% 波形样本协方差（实际优化时改变）
R_Phi_uncorr = (P/N) * eye(N);

% 接收与发射 steering vector 句柄（可自定义间距）
% 间距以波长为单位，默认半波长
d_tx = 0.5;   % 发射间距，后续在具体图中可以改变
d_rx = 0.5;
tx_steer = @(theta) exp(1j*2*pi*d_tx*(0:N-1)'*sind(theta));
rx_steer = @(theta) exp(1j*2*pi*d_rx*(0:M-1)'*sind(theta));

% 干扰加噪声协方差矩阵 Q (M x M)
Q = sigma2 * eye(M) + jammer_power * (rx_steer(jammer_angle) * rx_steer(jammer_angle)');

% fig2_main.m
% 模拟一个距离-角度成像场景，包含10个运动目标、强杂波、干扰机
% 使用不同自适应方法生成图像
% fig2.m
% 角度-距离成像示意（使用 Capon 方法）


% 模拟场景参数
num_range_bins = 20;          % 距离门数量
theta_grid = -90:0.5:90;      % 角度网格
num_angles = length(theta_grid);

% 随机放置 5 个点目标
num_targets = 5;
target_ranges = randi(num_range_bins, num_targets, 1);
target_angles = -60 + 120*rand(num_targets,1);
target_amps = ones(num_targets,1);

% 杂波：每个距离门有一个 -10° 的强杂波
clutter_angle = -10;
clutter_amp = 20;

% 生成每个距离门的接收数据
X_cell = cell(num_range_bins, 1);
for r = 1:num_range_bins
    % 目标回波
    S = zeros(M, L);
    idx = find(target_ranges == r);
    for k = idx'
        a = rx_steer(target_angles(k));
        v = tx_steer(target_angles(k));
        b = target_amps(k);
        S = S + a * b * (v.' * Phi);
    end
    % 杂波回波
    a_c = rx_steer(clutter_angle);
    v_c = tx_steer(clutter_angle);
    S = S + clutter_amp * a_c * (v_c.' * Phi);
    
    % 噪声+干扰（空域有色）
    Q_sqrt = chol(Q, 'lower');
    Z = Q_sqrt * (randn(M, L) + 1j*randn(M, L)) / sqrt(2);
    
    X_cell{r} = S + Z;
end

% Capon 谱估计
P_capon = zeros(num_range_bins, num_angles);
for r = 1:num_range_bins
    X = X_cell{r};
    R_hat = (X * X') / L;   % 样本协方差
    invR = inv(R_hat);
    for ia = 1:num_angles
        a = rx_steer(theta_grid(ia));
        P_capon(r, ia) = 1 / real(a' * invR * a);
    end
end

% 绘图
figure;
imagesc(theta_grid, 1:num_range_bins, 10*log10(abs(P_capon)));
xlabel('Angle (deg)'); ylabel('Range Bin');
title('Capon Angle-Range Image (simplified)');
colorbar;
caxis([-10 40]);
saveas(gcf, 'fig2.png');