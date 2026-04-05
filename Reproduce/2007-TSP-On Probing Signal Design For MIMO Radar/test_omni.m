

clc;
clear all;
close all;
addpath('./functions');

rng(42); 

%% 参数设置
M = 10;                     % 天线数（提高分辨率，使三峰分离）
N = 512;                    % 快拍数
c = ones(M, 1) * 1/M;             % 发射功率常数
% c = rand(M, 1);
% 目标
theta_targets = [-40, 0, 40];   % 度
beta = [1, 1, 1];

% 角度扫描
theta_grid = -90:0.1:90;
L = length(theta_grid);

% 导向矢量函数 (均匀线阵，半波长间距)
afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));

A = zeros(M, length(theta_targets));
for k = 1:length(theta_targets)
    A(:,k) = afun(theta_targets(k));
end

%% Capon MSE with the optimal beampattern matching design
%% 1 生成发射信号
x = sqrt(1/M) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);
%% 生成目标回波
y_target = zeros(M, N);
for k = 1:length(theta_targets)
    a = afun(theta_targets(k));
    y_target = y_target + beta(k) * a * (a.' * x);
end
%% 生成噪声
sigma2_dB = -20;
sigma2 = 10^(-sigma2_dB/10);
noise = sqrt(sigma2) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);

%% 总接收信号
y = y_target + noise;
% 样本协方差矩阵
Rxx = (x * x') / N;
Ryy = (y * y') / N;
Ryx = (y * x') / N;
Capon_omni = zeros(L, 1);
%% 扫描每个角度
% ========== Capon 空间谱, 严格按照公式(36) ==========
for idx = 1:L
    theta = theta_grid(idx);
    a = afun(theta);
    num_capon = a' / Ryy * Ryx * conj(a);
    denom_capon = (a' / Ryy * a) * (a.' * Rxx * conj(a));
    % Capon 谱值
    Capon_omni(idx) = (num_capon) / (denom_capon);   % 通常取模平方，保证为正
end
Capon_omni = abs(Capon_omni);
Capon_omni_norm = Capon_omni / max(Capon_omni);
[peaks, locs] = findpeaks(Capon_omni_norm, 'MinPeakHeight', 0.7*max(Capon_omni_norm), 'MinPeakDistance', 5);

%% 
figure(1);

plot(theta_grid, Capon_omni, 'r-', 'LineWidth', 1.5); hold on;
plot(theta_grid(locs), Capon_omni(locs), 'bo', 'MarkerSize', 8); hold on;
xlabel('\theta (degrees)');
ylabel('Capon spectrum (dB)');
legend('omnidirectional beampattern');
grid on; 
% xlim([-90, 90]);
% ylim([-90, 90]);








