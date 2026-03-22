


clc;
clear all;
close all;

rng(42); 
addpath('./functions');

clear; clc; close all;

%% 参数设置（与论文一致，但提高分辨率）
M = 20;                     % 天线数（提高分辨率，使三峰分离）
N = 256;                    % 快拍数
c = 1;                      % 发射功率常数
sigma2_dB = -10;            % 噪声功率 (dB)
sigma2 = 10^(sigma2_dB/10);
jammer_power = 1e6;         % 干扰功率 (60 dB)

% 目标
theta_targets = [-60, 0, 60];   % 度
beta = [1, 1, 1];

% 干扰
theta_jammer = 20;              % 度

% 角度扫描
theta_scan = -90:0.1:90;
L = length(theta_scan);

% 导向矢量函数 (均匀线阵，半波长间距)
a_func = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));
ac_func = @(theta) conj(a_func(theta));

%% 生成发射信号 x(n) ~ CN(0, (c/M)*I)

x = sqrt(c/M) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);

%% 生成目标回波
y_target = zeros(M, N);
for k = 1:length(theta_targets)
    a = a_func(theta_targets(k));
    ac = ac_func(theta_targets(k));
    y_target = y_target + beta(k) * conj(a) * (a' * x);
end

%% 生成干扰回波
jam = sqrt(jammer_power) * (randn(1, N) + 1j*randn(1, N)) / sqrt(2);
ac_jammer = ac_func(-theta_jammer);
y_jammer = ac_jammer * jam;

%% 生成噪声
w = sqrt(sigma2) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);

%% 总接收信号
y = y_target + y_jammer + w;

%% 样本协方差矩阵
Rxx = (x * x') / N;
Ryy = (y * y') / N;
Ryx = (y * x') / N;

%% 预计算 Ryy 的逆（用于 GLRT 分子和 Capon）
inv_Ryy = inv(Ryy);

%% 存储结果
GLRT = zeros(L, 1);
Capon = zeros(L, 1);

%% 扫描每个角度
for idx = 1:L
    theta = theta_scan(idx);
    a = a_func(theta);       % 发射导向矢量 M×1
    ac = ac_func(theta);     % 接收导向矢量 M×1

    % ========== GLRT 严格按公式(29)-(31)计算 ==========
    % 分子: a^* Ryy^{-1} a
    num = a' / Ryy * a;
    % 分母: a^* Rxx a
    denom = a.' * Rxx * ac;
    if denom < 1e-12
        GLRT(idx) = NaN;
        continue;
    end
    % 构造 Q = Ryy - (Ryx * a * a' * Ryx') / denom
    Q = Ryy - (Ryx * ac * a.' * Ryx') / denom;

    % 对 Q 求逆，加入极小正则化确保数值稳定
    reg = 1e-10 * trace(Q) / M;
    inv_Q = inv(Q + reg * eye(M));

    % 分母项: a^* Q^{-1} a
    denom_GLRT = a' / Q * a;

    GLRT(idx) = 1 - num / denom_GLRT;
    % ========== Capon 谱 ==========
    Capon(idx) = 1 / (a.'/ Ryy * ac);

    % % ========== Capon 空间谱 —— 严格按照公式(36) ==========
    % % 分子: a^* Ryy^{-1} Ryx ac
    % num_capon = a' / Ryy * Ryx * ac;   % 标量
    % % 分母: a^* Ryy^{-1} a  *  a^T Rxx ac
    % denom_capon = (a' / Ryy * a) * (a.' * Rxx * ac);
    % % Capon 谱值
    % Capon(idx) = abs(num_capon) / denom_capon;   % 通常取模平方，保证为正

end


%% 绘图
% figure(1);
% 
% plot(theta_scan,  pow2db(Capon/max(Capon)) , 'b-', 'LineWidth', 1.5); hold on;
% plot(theta_scan, pow2db(abs(GLRT)/max(abs(GLRT))) , 'r--', 'LineWidth', 1.5); hold on;
% xlabel('\theta (degrees)');
% ylabel('spectrum (dB)');
% grid on; 
% xlim([-90, 90]);
% legend('Capon spectrum (dB)', 'GLRT spectrum (dB)' );


figure(2);
% (a) Capon 空间谱 (dB)
subplot(1,2,1);
plot(theta_scan, pow2db(Capon/max(Capon)), 'b-', 'LineWidth', 1.5);
xlabel('\theta (degrees)');
ylabel('Capon spectrum (dB)');
title('(a) Capon');
grid on; xlim([-90, 90]);

% (b) GLRT 伪谱
subplot(1,2,2);
plot(theta_scan, pow2db(abs(GLRT)/max(abs(GLRT))), 'r-', 'LineWidth', 1.5);
xlabel('\theta (degrees)');
ylabel('\tilde{\phi}(\theta)');
title('(b) GLRT');
grid on; xlim([-90, 90]);