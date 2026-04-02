

clc;
clear all;
close all;

addpath('./functions');

rng(42); 


%% 参数设置 
M = 10;                     % 天线数（提高分辨率，使三峰分离）
N = 256;                    % 快拍数
c = ones(M,1);                      % 发射功率常数

% 目标
theta_targets = [-40, 0, 40];   % 度
beta = [1, 1, 1];

% 角度扫描
theta_grid = -90:0.1:90;
L = length(theta_grid);

% 导向矢量函数 (均匀线阵，半波长间距)
a_func = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));
ac_func = @(theta) conj(a_func(theta));

Delta = 5;
P_des = zeros(size(theta_grid));
% Desired beam pattern
idx = false(size(theta_grid));
for i = 1:numel(theta_targets)
    idx = idx | theta_grid >= theta_targets(i)-Delta & theta_grid <= theta_targets(i)+Delta;
end
P_des(idx) = 1;
L = length(theta_grid);
 
% 权重
w_l = ones(L, 1);           % 所有网格点权重相同
wc = 1;                    % 交叉项权重（可调）
[R_opt, alpha, ~ ] = BeampatternMatchingDesign(c, M, w_l, wc, theta_targets, theta_grid, P_des);
Rsqrt =  sqrtm(R_opt); % R^(0.5);

A = zeros(M, L);
for i = 1 : length(theta_grid)
    theta = theta_grid(i);
    a = a_func(theta);       % 发射导向矢量 M×1
    % ac = ac_func(theta);     % 接收导向矢量 M×1
    A(:,i) = a;
end

sigma2_dB = -20:5:20;               % 噪声功率 (dB)
res = zeros(2, length(sigma2_dB));
Iters = 100;
for i = 1:length(sigma2_dB)
    sigma2 = 10^(-sigma2_dB(i)/10);
    for it = 1:Iters
        %%  Capon MSE with initial omnidirectional probing
        %% 1 生成发射信号 x(n) ~ CN(0, (c/M)*I)
        x = sqrt(c/M) .* (randn(M, N) + 1j*randn(M, N)) / sqrt(2);
        %% 生成目标回波
        y_target = zeros(M, N);
        for k = 1:length(theta_targets)
            a = a_func(theta_targets(k));
            y_target = y_target + beta(k) * a * (a.' * x);
        end
        %% 生成噪声
        w = sqrt(sigma2) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);
        %% 总接收信号
        y = y_target + w;
        % 样本协方差矩阵
        Rxx = (x * x') / N;
        Ryy = (y * y') / N;
        Ryx = (y * x') / N;
        Capon_omni = zeros(L, 1);
        %% 扫描每个角度
        for idx = 1:L
            theta = theta_grid(idx);
            a = a_func(theta);       % 发射导向矢量 M×1
            ac = ac_func(theta);     % 接收导向矢量 M×1
            % ========== Capon 空间谱 —— 严格按照公式(36) ==========
            % 分子: a^* Ryy^{-1} Ryx ac
            num_capon = a' / Ryy * Ryx * ac;   % 标量
            % 分母: a^* Ryy^{-1} a  *  a^T Rxx ac
            denom_capon = (a' / Ryy * a) * (a.' * Rxx * ac);
            % Capon 谱值
            Capon_omni(idx) = abs(num_capon) / abs(denom_capon);   % 通常取模平方，保证为正
        end
        Capon_omni_norm = Capon_omni / max(Capon_omni);
        [~, locs] = findpeaks(Capon_omni_norm, theta_grid, 'MinPeakHeight', 0.1*max(Capon_omni_norm), 'MinPeakDistance', 5);
        % disp('Capon 峰值角度：');
        % disp(locs);
        res(1, i) = res(1, i) + (theta_targets(1) - locs(1))^2;

        %% AML MSE with probing using the beampattern matching design
        %% 1 生成发射信号 
        w = (randn(M, N) + 1j * randn(M, N)) / sqrt(2);
        x = Rsqrt * w;
        %% 生成目标回波
        y_target = zeros(M, N);
        for k = 1:length(theta_targets)
            a = a_func(theta_targets(k));
            y_target = y_target + beta(k) * a * (a.' * x);
        end
        %% 生成噪声
        w = sqrt(sigma2) * (randn(M, N) + 1j*randn(M, N)) / sqrt(2);
        %% 总接收信号
        y = y_target + w;
        % 样本协方差矩阵
        Rxx = (x * x') / N;
        Ryy = (y * y') / N;
        Ryx = (y * x') / N;
        Capon_optim = zeros(L, 1);
        %% 扫描每个角度
        for idx = 1:L
            theta = theta_grid(idx);
            a = a_func(theta);       % 发射导向矢量 M×1
            ac = ac_func(theta);     % 接收导向矢量 M×1
            % ========== Capon 空间谱 —— 严格按照公式(36) ==========
            % 分子: a^* Ryy^{-1} Ryx ac
            num_capon = a' / Ryy * Ryx * ac;   % 标量
            % 分母: a^* Ryy^{-1} a  *  a^T Rxx ac
            denom_capon = (a' / Ryy * a) * (a.' * Rxx * ac);
            % Capon 谱值
            Capon_optim(idx) = abs(num_capon) / abs(denom_capon);   % 通常取模平方，保证为正
        end
        Capon_optim_norm = Capon_optim / max(Capon_optim);
        [peaks, locs] = findpeaks(Capon_optim_norm, theta_grid, 'MinPeakHeight', 0.1*max(Capon_optim_norm), 'MinPeakDistance', 5);
        % disp('Capon 峰值角度：');
        % disp(locs);
        res(2, i) = res(2, i) + (theta_targets(1) - locs(1))^2;

    end
end
%% 
figure(1);
% (a) Capon 空间谱 (dB)
subplot(1,2,1);
plot(theta_grid, Capon_omni_norm, 'b-', 'LineWidth', 1.5); hold on;
xlabel('\theta (degrees)');
ylabel('Capon spectrum (dB)');
legend('omnidirectional beampattern');
grid on; 
xlim([-90, 90]);
% (b) Capon 空间谱 (dB)
subplot(1,2,2);
plot(theta_grid, Capon_optim_norm, 'r-', 'LineWidth', 1.5);
xlabel('\theta (degrees)');
ylabel('Capon spectrum (dB)');
legend('Optimized beampattern');
grid on; xlim([-90, 90]);

figure(2);
semilogy(sigma2_dB, res(1,:)/Iters, 'b-', 'LineWidth', 1.5); hold on;
semilogy(sigma2_dB, res(2,:)/Iters, 'r-', 'LineWidth', 1.5); hold on;
xlabel('Noise level~(dB)');
ylabel('MSE');
legend('omnidirectional beampattern', 'Optimized beampattern');
grid on;

ylim([-90, 90]);
























































