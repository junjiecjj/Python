

% https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247489724&idx=1&sn=ae3753e44cd49e9e3d9de78461c72cd3&chksm=ce4557c1bc1a28aa76ddeb920ca433b564c31db0abb4998af426fda4733e914bcd916a433c84&mpshare=1&scene=1&srcid=01272cWvizqv9l5hsxW3QoJT&sharer_shareinfo=333a8953bc257a68c4e09fa24fe8563d&sharer_shareinfo_first=333a8953bc257a68c4e09fa24fe8563d&exportkey=n_ChQIAhIQW2SGDmFpWCYRevIbypoNRRKfAgIE97dBBAEAAAAAAJK4OnYUb24AAAAOpnltbLcz9gKNyK89dVj0tCQXP2Hhpo7iz6j5i0c07QwYsvCHy16Sf46nvqMDxH6tWaxVFW0sq6Bjn1tCzzrQN4EmsdEfHnJMOBJVYBbB%2FscngQR4AK%2BSc8iGpNKHYxQMb7lcbhG31YvhbLYOjxbsOAMUXpBwt8O4ltFcD9h%2FTghbpcVDcvUqNHkHGyYcnNH7hCOWqn4ECmNRmylZFxELdCaFm4Wo97l%2BQoijNfUpknGC8dX9Dy4zpnNUpWnoRj0LCzgjph5Ng8jr4ChF%2BL2dkJoCZtbOKG7avsq5sewwklk%2Bf0sCEcKb5YX7m2FgyaNGOwQAQov35gFvBHdxT6BdIKsYSUqL4xqw&acctmode=0&pass_ticket=huAD56jGGJNUxgP9FB3MViFyKo4KF4nG6ahU8PGJ%2ByubhYTP1PQGQ%2F0evhcDbcd8&wx_header=0#rd


%% 7阵元线阵 DFT 15个和/差波束形成仿真
clear;clc;close all;
%% 1. 参数设置
fc = 10e9;          % 载波频率 10GHz
lambda = 3e8/fc;    % 波长
d = lambda/2;       % 阵元间距
N_elem = 7;         % 阵元数
N_beam = 15;        % 波束数
theta_range = [-60:0.1:60]; % 扫描角度范围(°)
theta_beam = linspace(-60,60,N_beam); % 15个波束指向角(°)
%% 2. 构造过采样DFT权值矩阵
W = zeros(N_elem,N_beam);
for m = 0:N_beam-1
    for n = 0:N_elem-1
        W(n+1,m+1) = (1/sqrt(N_elem)) * exp(-1j*2*pi*m*n/N_beam);
    end
end
%% 3. 构造和/差波束权值
% 和波束权值：DFT权值 × 全1向量（同相加权）
w_sum = W .* ones(N_elem,N_beam);
% 差波束权值：DFT权值 × 线性斜率因子（对称反相）
diff_factor = [-3,-2,-1,0,1,2,3];
w_diff = W.*(diff_factor' * ones(1,N_beam));
%% 4. 计算阵列流形与波束方向图
AF_sum = zeros(N_beam,length(theta_range)); % 和波束方向图
AF_diff = zeros(N_beam,length(theta_range)); % 差波束方向图
k = 2*pi/lambda;
for m = 1:N_beam
    for i = 1:length(theta_range)
        theta = theta_range(i)*pi/180;
        % 阵列流形向量
        a = exp(1j*k*d*(0:N_elem-1)'*sin(theta));
        % 计算和/差波束方向图
        AF_sum(m,i) = abs(w_sum(:,m)' * a);
        AF_diff(m,i) = abs(w_diff(:,m)' * a);
    end
    % 归一化
    AF_sum(m,:) = AF_sum(m,:)/max(AF_sum(m,:));
    AF_diff(m,:) = AF_diff(m,:)/max(AF_diff(m,:));
end
%% 5. 绘图
for i = 1:N_elem
    figure(1)
    plot(theta_range,AF_sum);
    % plot(theta_range,20*log10(AF_sum));
    title('15个和波束方向图');
    xlabel('角度 (°)');ylabel('归一化增益 (dB)');
    grid on;
    figure(2)
    plot(theta_range,AF_diff);
    % plot(theta_range,20*log10(AF_diff));
    title('15个差波束方向图');
    xlabel('角度 (°)');ylabel('归一化增益 (dB)');
    grid on;
end
figure(4)
plot(theta_range,AF_sum(1,:));
xlabel('角度 (°)');ylabel('增益 (dB)');
title('单个和波束方向图');
grid on;
figure(3)
plot(theta_range,AF_diff(1,:));
xlabel('角度 (°)');ylabel('增益 (dB)');
title('单个差波束方向图');
grid on;


%% 7阵元线阵波束形成仿真
clear; clc; close all;
%% 1. 系统参数设置
N = 7;                  % 阵元数
M = 16;                 % 波束数量
theta_target = 20.4630; % 目标方位角（度）
theta_range = linspace(-45, 45, M); % 16个波束指向角度
lambda = 1;             % 波长（归一化）
d = lambda/2;           % 阵元间距（半波长）
s0 = 1;                 % 目标信号复振幅
%% 2. 构造目标导向向量
theta0_rad = deg2rad(theta_target);
a_target = exp(1j * 2*pi*d/lambda * (0:N-1)' * sin(theta0_rad));
%% 3. 构造16个波束的权值矩阵
W = zeros(N, M);
for m = 1:M
    theta_m_rad = deg2rad(theta_range(m));
    W(:,m) = exp(1j * 2*pi*d/lambda * (0:N-1)' * sin(theta_m_rad));
end
%% 4. 生成不同信噪比的回波信号
% 无噪声回波
x_noiseless = a_target * s0;
% SNR=23dB 回波
snr1 = 23;
x_snr23 = awgn(x_noiseless, snr1, 'measured');
% SNR=10dB 回波
snr2 = 10;
x_snr10 = awgn(x_noiseless, snr2, 'measured');
%% 5. 波束形成计算
y_noiseless = abs(conj(W') * x_noiseless);
y_snr23 = abs(conj(W') * x_snr23);
y_snr10 = abs(conj(W') * x_snr10);
%% 6. 绘图
figure('Color','w');
% 无噪声
plot(1:M, y_noiseless/max(y_noiseless), 'LineWidth',1.5);
title('当不加噪声时，有信号的距离门上16个波束输出');
xlabel('波束通道'); ylabel('幅度');
xlim([1 M]); ylim([0 1]);
grid on;
% SNR=23dB
figure('Color','w');
plot(1:M, y_snr23/max(y_snr23), 'LineWidth',1.5);
title('SNR=23dB时，有信号的距离门上16个波束输出');
xlabel('波束通道'); ylabel('幅度');
xlim([1 M]); ylim([0 1]);
grid on;
% SNR=10dB
figure('Color','w');
plot(1:M, y_snr10/max(y_snr10), 'LineWidth',1.5);
title('当SNR=10dB时，有信号的距离门上16个波束输出');
xlabel('波束通道'); ylabel('幅度');
xlim([1 M]); ylim([0 1]);
grid on;
% sgtitle('7阵元线阵16波束形成输出对比','FontSize',14);


