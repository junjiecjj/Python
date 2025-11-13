

clear; close all; clc;
addpath('/home/jack/cvx-a64/cvx');

cvx_setup;

%% 全局参数设置
global PT_global;

% 基本系统参数（与论文完全一致）
L = 1024;
d = 0.5;
PT_dBm = 40;       
PT = 10^((PT_dBm)/10) * 1e-3;
PT_global = PT;

sigma2_c = 0.001;  % 0 dBm = 0.001 W  
sigma2_r = 0.001;  % 0 dBm = 0.001 W
mu = 1;             % 莱斯因子

fprintf('\n=== 实验1：收敛性能分析 (图2) ===\n');

% 系统参数定义（严格按照论文设置）
L = 1024;           % 信号长度
d = 0.5;           % 天线间距 (λ/2)
mu = 1;            % 莱斯因子 μ=1

% 功率设置（论文标准）
PT_dBm = 40;       % 发射功率 40 dBm
PT_watts = 10;  % 转换为瓦特：40dBm = 10W

% 噪声功率设置（论文标准）
noise_dBm = 0;     % 0 dBm噪声功率
sigma2_c = 0.001;  % 0dBm = 0.001W
sigma2_r = sigma2_c;

fprintf('系统参数确认:\n');
fprintf('  发射功率: %d dBm = %.2f W\n', PT_dBm, PT_watts);
fprintf('  噪声功率: %d dBm = %.6f W\n', noise_dBm, sigma2_c);
fprintf('  SNR: %.1f dB\n', PT_dBm - noise_dBm);































































