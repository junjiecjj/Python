

clc;
clear all;
close all;
addpath('./functions');

rng(42); 

%% 参数设置
M = 10;                     % 天线数（提高分辨率，使三峰分离）
N = 256;                    % 快拍数
c = ones(M, 1) * 1/M;             % 发射功率常数
% c = rand(M, 1);
% 目标
theta_targets = [-40, 0, 3];   % 度
beta = [1, 1, 1];

% 干扰
jammer_power = 25;         % 干扰功率 (60 dB)
jammer_power = 10^(jammer_power/10);
theta_jammer = 25;              % 度

% noise
sigma2_dB = -20;
sigma2 = 10^(sigma2_dB /10);
% 生成噪声
noise = sqrt(sigma2) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);

% 角度扫描
theta_grid = -90:0.1:90;
L = length(theta_grid);

% 导向矢量函数 (均匀线阵，半波长间距)
afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));

% A = zeros(M, length(theta_targets));
% for k = 1:length(theta_targets)
%     A(:, k) = afun(theta_targets(k));
% end

%% initial omnidirectional probing
%  Capon MSE with initial omnidirectional probing
% 1 生成发射信号 x(n) ~ CN(0, (c/M)*I)
x = sqrt(c) .* (randn(M, N) + 1j*randn(M, N)) / sqrt(2);
% 生成目标回波
X = zeros(M, N);
for k = 1:length(theta_targets)
    ak = afun(theta_targets(k));
    X = X + beta(k) * ak * (ak.' * x);
end

% 生成干扰回波
jam = sqrt(jammer_power) * (randn(1, N) + 1j*randn(1, N)) / sqrt(2);
ac_jammer = afun(theta_jammer);
y_jammer = ac_jammer * jam;

% 总接收信号
y = X + noise + y_jammer * 0;
% 样本协方差矩阵
Rxx = (x * x') / N;
Ryy = (y * y') / N;
Ryx = (y * x') / N;

[Capon_omni, peaks, theta_est] = CaponBeampatterns(M, Rxx, Ryx, Ryy, theta_grid);
[GLRT_omni, peaks1, theta_est1] = GLRTBeampatterns(M, Rxx, Ryx, Ryy, theta_grid);

%% the optimal beampattern matching
% Desired beam pattern
Theta_est = [-40, 1.5];
Delta = 10;
P_des = zeros(size(theta_grid));
idx = false(size(theta_grid));
for i = 1:numel(Theta_est)
    idx = idx | theta_grid >= Theta_est(i)-Delta & theta_grid <= Theta_est(i)+Delta;
end
P_des(idx) = 1;

% 权重
w_l = ones(L, 1);          % 所有网格点权重相同
wc = 1;                    % 交叉项权重（可调）
[R_opt, alpha, ~ ] = BeampatternMatchingDesign(c, M, w_l, wc, Theta_est, theta_grid, P_des);

P_opt1 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = afun(theta_grid(i));
    P_opt1(i) = real(a_theta' * R_opt * a_theta);
end

% 1 生成发射信号 x(n) ~ CN(0, R_opt)
% w = (randn(M, N) + 1j * randn(M, N)) / sqrt(2);
% x = sqrtm(R_opt) * w;
x = WaveformSynthesisXoptimR(N, R_opt, 2);
% 生成目标回波
X = zeros(M, N);
for k = 1:length(theta_targets)
    ak = afun(theta_targets(k));
    X = X + beta(k) * ak * (ak.' * x);
end

% 总接收信号
y = X + noise + y_jammer * 0;
% 样本协方差矩阵
Rxx = (x * x') / N;
Ryy = (y * y') / N;
Ryx = (y * x') / N;

[Capon_optim, peaks2, theta_est2] = CaponBeampatterns(M, Rxx, Ryx, Ryy, theta_grid);
[GLRT_optim, peaks3, theta_est3] = GLRTBeampatterns(M, Rxx, Ryx, Ryy, theta_grid);

%% 
figure(1);
% (a) Capon 空间谱 (dB)
subplot(2,2,1);
plot(theta_grid, Capon_omni, 'b-', 'LineWidth', 1.5); hold on;
% plot(theta_est, peaks, 'ro', 'MarkerSize', 8); hold on;
xlabel('\theta (degrees)');
ylabel('Capon spectrum (dB)');
legend('Capon omni beampattern');
grid on; 
xlim([-90, 90]);
% (b) GLRT 空间谱 (dB)
subplot(2,2,2);
plot(theta_grid, GLRT_omni, 'b-', 'LineWidth', 1.5); hold on;
% plot(theta_est1, peaks1, 'ro', 'MarkerSize', 8); hold on;
xlabel('\theta (degrees)');
ylabel('GLRT spectrum (dB)');
legend('GLRT omni beampattern');
grid on; 

subplot(2,2,3);
plot(theta_grid, Capon_optim, 'b-', 'LineWidth', 1.5); hold on;
% plot(theta_est2, peaks2, 'ro', 'MarkerSize', 8); hold on;
xlabel('\theta (degrees)');
ylabel('Capon spectrum (dB)');
legend('Capon Optimized beampattern');
grid on; 
xlim([-90, 90]);
% (b) GLRT 空间谱 (dB)
subplot(2,2,4);
plot(theta_grid, GLRT_optim, 'b-', 'LineWidth', 1.5); hold on;
% plot(theta_est3, peaks3, 'ro', 'MarkerSize', 8); hold on;
xlabel('\theta (degrees)');
ylabel('GLRT spectrum (dB)');
legend('GLRT Optimized beampattern');
grid on; 

%% 可选：绘制发射波束图对比
% figure(2);
% plot(theta_grid, abs(P_des * alpha), 'k--', 'LineWidth', 1.5); hold on;
% plot(theta_grid, P_opt1, 'r-', 'LineWidth', 1.5); hold on;
% 
% xlabel('\theta (degrees)');
% ylabel('Beampattern');
% legend('Desired', 'Optimized,w_c=1' );
% title('Transmit Beampattern');
% grid on;
% 




















































