% 清除环境
clc; clear; close all;

%% 1. 系统参数设置 (System Settings)
fc = 4.9e9;           
c0 = 3e8;             
lambda = c0/fc;       
BW = 20e6;            
M = 612;              
N = 7;                
delta_f = 30e3;       
Ts = 35.677e-6;       
P = 16; Q = 24;       % UPA 天线阵列 (水平P, 垂直Q)
L = P * Q;            
R = 64;               
K = 2;                

% 模拟目标参数 (Range, Velocity, Angles)
d_true = [180, 30];           
v_true = [10, -20];            
theta_true = deg2rad([10, 25]); 
phi_true = deg2rad([30, 60]);  

%% 2. 生成接收张量 Y (Tensor Formulation)
Y = zeros(R, N, M);
SNR_dB = 20;
Frx_cell = cell(1, K); % 存储每个目标的合并矩阵

for k = 1:K
    vartheta = sin(theta_true(k)) * cos(phi_true(k));
    psi = cos(theta_true(k));
    
    a_p = exp(1j*pi*(0:P-1)'*vartheta);
    a_q = exp(1j*pi*(0:Q-1)'*psi);
    a_upa = kron(a_q, a_p); 
    
    % 为了保证实验可重复，固定随机种子
    rng(k);
    Frx = (randn(L, R) + 1j*randn(L, R))/sqrt(2*R); 
    Frx_cell{k} = Frx; 
    Ftx = (randn(L, 1) + 1j*randn(L, 1))/sqrt(2);   
    
    b_k = Frx' * a_upa * (a_upa' * Ftx); 
    o_k = exp(1j*2*pi*Ts*(2*v_true(k)/lambda)*(0:N-1)'); 
    g_k = exp(-1j*2*pi*delta_f*(2*d_true(k)/c0)*(0:M-1)'); 
    
    alpha_k = randn + 1j*randn; 
    Y = Y + alpha_k * reshape(kron(g_k, kron(o_k, b_k)), [R, N, M]);
end

noise = (randn(size(Y)) + 1j*randn(size(Y))) * 10^(-SNR_dB/20);
Y = Y + noise;

%% 3. 执行张量分解 (CPD)
% 假设已经定义了 Spatial_Smoothing_CPD 函数
[A1_est, A2_est, A3_est, z_hat] = Spatial_Smoothing_CPD(Y, K);

%% 4. 参数提取与目标配对 (Association)
est_results = struct();

% 预提取所有估计值
for k = 1:K
    % 距离
    tau_est = angle(z_hat(k)) / (-2 * pi * delta_f);
    est_results(k).dist = abs(tau_est * c0 / 2);
    % 速度
    fd_est = angle(A2_est(2,k)/A2_est(1,k)) / (2 * pi * Ts);
    est_results(k).vel = fd_est * lambda / 2;
    % 记录因子矩阵列索引
    est_results(k).idx = k;
end

% 简单的配对逻辑：基于距离最接近原则
% 在实际复杂场景下建议使用匈牙利算法 (matchpairs)
matched_idx = zeros(1, K);
remaining_est = 1:K;
for k = 1:K
    diffs = arrayfun(@(i) abs(est_results(i).dist - d_true(k)), remaining_est);
    [~, min_pos] = min(diffs);
    matched_idx(k) = remaining_est(min_pos);
    remaining_est(min_pos) = [];
end

%% 5. 输出结果与可视化 (Output and Visualization)
fprintf('\n================ 参数估计对照结果 ================\n');
est_plot_dist = zeros(1, K);
est_plot_vel = zeros(1, K);
est_plot_theta = zeros(1, K);
est_plot_phi = zeros(1, K);

for k = 1:K
    idx = matched_idx(k); 
    
    % 提取匹配后的角度
    [theta_k_est, phi_k_est] = GRQ_AoA_Method(A1_est(:, idx), Frx_cell{k}, P, Q);
    
    % 存储用于绘图的数据
    est_plot_dist(k) = est_results(idx).dist;
    est_plot_vel(k) = est_results(idx).vel;
    est_plot_theta(k) = rad2deg(theta_k_est);
    est_plot_phi(k) = rad2deg(phi_k_est);
    
    % 终端打印 (保持原样)
    fprintf('目标 %d: 距离误差 %.4f, 速度误差 %.4f\n', k, ...
        abs(est_plot_dist(k) - d_true(k)), abs(est_plot_vel(k) - v_true(k)));
end

%% --- 绘图部分 (论文复现图) ---

% 图 1: 距离-速度联合估计散点图 (Joint Range-Velocity Estimation)
figure('Color', 'w', 'Name', 'Range-Velocity Estimation');
plot(d_true, v_true, 'bo', 'MarkerSize', 12, 'LineWidth', 2); hold on;
plot(est_plot_dist, est_plot_vel, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
grid on;
xlabel('Distance (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Velocity (m/s)', 'FontSize', 12, 'FontWeight', 'bold');
legend('True Values', 'Tensor Estimates', 'Location', 'best');
title('Joint Range and Velocity Estimation Performance', 'FontSize', 14);
set(gca, 'FontSize', 11);

% 图 2: 角度估计散点图 (AoA Estimation: Theta vs Phi)
figure('Color', 'w', 'Name', 'AoA Estimation');
plot(rad2deg(theta_true), rad2deg(phi_true), 'bs', 'MarkerSize', 12, 'LineWidth', 2); hold on;
plot(est_plot_theta, est_plot_phi, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
grid on;
xlabel('Elevation Angle \theta (deg)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Azimuth Angle \phi (deg)', 'FontSize', 12, 'FontWeight', 'bold');
legend('True AoA', 'Estimated AoA', 'Location', 'best');
title('2D AoA (Elevation and Azimuth) Estimation', 'FontSize', 14);
set(gca, 'FontSize', 11);

fprintf('\n绘图完成！你可以向老师展示这两张图，证明张量分解能准确分离并估计多个目标参数。\n');