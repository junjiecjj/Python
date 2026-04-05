%% MUSIC 算法 DOA 估计（MATLAB 实现）
clear; clc; close all;
rng(42); 
%% 参数设置
M = 8;                      % 阵元个数
K = 3;                      % 信源数目
doa_deg = [-40, 0, 60];      % 真实波达方向（度）

f0 = 1e6;
f = [0.1, 0.2, 0.3] * f0;   % 各信号频率（保证正交）
snr = 30;                   % 信噪比 (dB)
N = 10240;                  % 快拍数
fs = 1e8;                   % 采样频率 (Hz)
Ts = 1/fs;
t = (0:N-1) * Ts;

%% 生成阵列接收信号 X (N × Ns)
% X = zeros(M, N);
% for i = 1:K
%     a_k = exp(-1j * pi * (0:M-1)' * sind(doa_deg(i)));   % 导向矢量（列向量）
%     s = exp(1j * 2 * pi * f(i) * t);        % 信号波形
%     % s = randn(1, N);
%     X = X + a_k * s;                        % 外积相加
% end

%% 生成阵列接收信号 X (M × Ns)
A = zeros(M, K);
for k = 1:K
    a_k = exp(1j * pi * (0:M-1)' * sind(doa_deg(k)));
    A(:,k) = a_k;
end
S = zeros(K, N);
for k = 1:K
    sk = exp(1j * 2 * pi * f(k) * t);
    % sk = randn(1, N);
    S(k, :) = sk;
end
X = A * S;

%% 添加高斯白噪声
Xpow = mean(abs(X).^2, 'all');              % 信号平均功率
noiseVar = Xpow * 10^(-snr/10);             % 噪声方差
Z = sqrt(noiseVar/2) * (randn(size(X)) + 1j*randn(size(X)));
Y = X + Z;

%% 计算协方差矩阵 R
Rss = S * S' / N;
Rxx = X * X' / N;  % == A*Rss*A'
Rzz = Z * Z' / N;
Ryy = Y * Y' / N;     % N×N 协方差矩阵

[VS, DS] = eig(Rss);
[VX, DX] = eig(Rxx);
[VZ, DZ] = eig(Rzz);
[VY, DY] = eig(Ryy);

%% 调用 MUSIC 算法
Thetalst = -90:0.5:90;
[Pmusic, angle_est, peaks] = MUSIC(Ryy, Thetalst, K, M);
[Pcapon, angle_est1, peaks1] = Capon(Ryy, Thetalst, M);

%% 绘制空间谱
figure(1);
plot(Thetalst, Pmusic, 'b-', 'LineWidth', 1.5); hold on;
plot(Thetalst, Pcapon, 'r--', 'LineWidth', 1.5); hold on;

% 标记真实角度
for i = 1:K
    xline(doa_deg(i), 'g--', 'LineWidth', 1);
end
% 标记估计角度

grid on;
xlabel('角度 (度)'); ylabel('空间谱 (dB)');
title('MUSIC 算法空间谱');

legend('MUSIC','Capon', '真实 DOA');
fprintf('估计的波达方向：\n');
disp(angle_est);
disp(angle_est1);

