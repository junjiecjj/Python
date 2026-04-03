

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
theta_targets = [-45, 0, 60];   % 度
beta = [1, 1, 1];

% 角度扫描
theta_grid = -90:0.1:90;
L = length(theta_grid);

% 导向矢量函数 (均匀线阵，半波长间距)
a_func = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));

A = zeros(M, length(theta_targets));
for k = 1:length(theta_targets)
    A(:,k) = a_func(theta_targets(k));
end

% Desired beam pattern
Delta = 5;
P_des = zeros(size(theta_grid));
idx = false(size(theta_grid));
for i = 1:numel(theta_targets)
    idx = idx | theta_grid >= theta_targets(i)-Delta & theta_grid <= theta_targets(i)+Delta;
end
P_des(idx) = 1;

% 权重
w_l = ones(L, 1);          % 所有网格点权重相同
wc = 1;                    % 交叉项权重（可调）
[R_opt, alpha, ~ ] = BeampatternMatchingDesign(c, M, w_l, wc, theta_targets, theta_grid, P_des);
Rsqrt =  sqrtm(R_opt);     % R^(0.5);

sigma2_dB = -20:5:20;               % 噪声功率 (dB)
res = zeros(2, length(sigma2_dB));
Iters = 100;
for i = 1:length(sigma2_dB)
    sigma2_dB(i)
    sigma2 = 10^(-sigma2_dB(i)/10);
    for it = 1:Iters
        %% 生成噪声
        noise = sqrt(sigma2) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);
        %%  Capon MSE with initial omnidirectional probing
        %% 1 生成发射信号 x(n) ~ CN(0, (c/M)*I)
        x = sqrt(1/M) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);
        %% 生成目标回波
        y_target = zeros(M, N);
        for k = 1:length(theta_targets)
            a = a_func(theta_targets(k));
            y_target = y_target + beta(k) * a * (a.' * x);
        end
        %% 总接收信号
        y = y_target + noise;
        % 样本协方差矩阵
        Rxx = (x * x') / N;
        Ryy = (y * y') / N;
        Ryx = (y * x') / N;
        Capon_omni = zeros(L, 1);
        %% 扫描每个角度
        % ========== Capon 空间谱 严格按照公式(36) ==========
        for idx = 1:L
            theta = theta_grid(idx);
            a = a_func(theta);
            num_capon = a' / Ryy * Ryx * conj(a);
            denom_capon = (a' / Ryy * a) * (a.' * Rxx * conj(a));
            % Capon 谱值
            Capon_omni(idx) = (num_capon) / (denom_capon);   % 通常取模平方，保证为正
        end
        Capon_omni = abs(Capon_omni);
        Capon_omni_norm = Capon_omni / max(Capon_omni);
        [~, locs] = findpeaks(Capon_omni_norm, theta_grid, 'MinPeakHeight', 0.5*max(Capon_omni_norm), 'MinPeakDistance', 5);
        res(1, i) = res(1, i) + (theta_targets(1) - locs(1))^2;

        %% Capon MSE with the optimal beampattern matching design
        %% 1 生成发射信号
        w = (randn(M, N) + 1j * randn(M, N)) / sqrt(2);
        x = Rsqrt * w;
        %% 生成目标回波
        y_target = zeros(M, N);
        for k = 1:length(theta_targets)
            a = a_func(theta_targets(k));
            y_target = y_target + beta(k) * a * (a.' * x);
        end
        %% 总接收信号
        y = y_target + noise;
        % 样本协方差矩阵
        Rxx = R_opt;
        Ryy = (y * y') / N;
        Ryx = (y * x') / N;
        Capon_optim = zeros(L, 1);
        %% 扫描每个角度
        % ========== Capon 空间谱, 严格按照公式(36) ==========
        for idx = 1:L
            theta = theta_grid(idx);
            a = a_func(theta);
            num_capon = a' / Ryy * Ryx * conj(a);
            denom_capon = (a' / Ryy * a) * (a.' * Rxx * conj(a));
            % Capon 谱值
            Capon_optim(idx) = (num_capon) / (denom_capon);   % 通常取模平方，保证为正
        end
        Capon_optim = abs(Capon_optim);
        Capon_optim_norm = Capon_optim / max(Capon_optim);
        [peaks, locs] = findpeaks(Capon_optim_norm, theta_grid, 'MinPeakHeight', 0.3*max(Capon_optim_norm), 'MinPeakDistance', 5);
        res(2, i) = res(2, i) + (theta_targets(1) - locs(1))^2;
    end
end

res = res/Iters;

%% 
figure(1);
% (a) Capon 空间谱 (dB)
subplot(1,2,1);
plot(theta_grid, Capon_omni, 'b-', 'LineWidth', 1.5); hold on;
xlabel('\theta (degrees)');
ylabel('Capon spectrum (dB)');
legend('omnidirectional beampattern');
grid on; 
xlim([-90, 90]);
% (b) Capon 空间谱 (dB)
subplot(1,2,2);
plot(theta_grid, Capon_optim, 'r-', 'LineWidth', 1.5);
xlabel('\theta (degrees)');
ylabel('Capon spectrum (dB)');
legend('Optimized beampattern');
grid on; 
% xlim([-90, 90]);
% ylim([-90, 90]);

figure(2);
semilogy(sigma2_dB, res(1,:), 'b-', 'LineWidth', 1.5); hold on;
semilogy(sigma2_dB, res(2,:), 'r-', 'LineWidth', 1.5); hold off;
xlabel('Noise level~(dB)');
ylabel('MSE');
legend('omnidirectional beampattern', 'Optimized beampattern');
grid on;

























































