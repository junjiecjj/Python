% https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247489027&idx=1&sn=5102d72841d0d4775a2c9b9ce9f50c36&chksm=ce8061c54bd097a9d8f0e8b9dd219b0b0d3224f46f8e632fe0ef1366258bcd66dc0ca4263483&mpshare=1&scene=1&srcid=01273K9q8gX5WatCG4WGGg2N&sharer_shareinfo=e63654ab97d5e99823b51f0e1505cd60&sharer_shareinfo_first=e63654ab97d5e99823b51f0e1505cd60&exportkey=n_ChQIAhIQHn5Tjx5A8ie%2FnBwSB0aAhRKUAgIE97dBBAEAAAAAAFVfGRkQKH8AAAAOpnltbLcz9gKNyK89dVj0bsCtn6i2PjfpmCjCrUASfYYmHqCFV6q4Qx5din2STQou%2BDNBRAJ1HWoLURrvsQW7R7p2mxPG%2BYAy9pB9GfmuSAKcAOrPsujbWqdRanCzmamTEdOdbqZiWq0hE79GluBmxgIrgi8YvrmDKFAeh2%2BSZOBhjew54O6NExYQaxfoHQbEeZdbn%2FWy%2BTUy9ShkfS8jn97%2FFP5NUft%2FheXj4wB16qLoNRQ2UF2KXJTDYmCwPFSnbgZV6UpMaQBdyKT24G68AnCk7moDFLtNASFEGazu0mJ1X12Lp8megxjxb%2FZ%2FuYfuNB1J0D8TD9zcfz4RJQ%3D%3D&acctmode=0&pass_ticket=PCeKSaBKhvCsyfFjwcr95LKe0Ky2owiBIz3fUsyldkAq0LB%2FxFOwIykLZXg%2FCp7F&wx_header=0#rd


% 仿真1：一次、二次固定对消仿真（常用的MTI）

clear; clc; close all;
%% ===================== 1. 系统参数设置 =====================
PRF = 1000;          % 脉冲重复频率 (Hz)
PRT = 1/PRF;         % 脉冲重复周期 (s)
N_pulse = 64;        % 相干脉冲数（慢时间长度）
fc = 10e9;           % 载波频率 (10GHz)
lambda = 3e8/fc;     % 雷达波长 (m)
v_target = 50;       % 目标径向速度 (m/s)
f_d = 2*v_target/lambda;  % 目标多普勒频率 (Hz)
CNR = 25;            % 杂波信噪比 (dB)
SNR = 10;            % 目标信噪比 (dB)
%% ===================== 2. 仿真雷达回波（慢时间序列） =====================
t_slow = (0:N_pulse-1)*PRT;  % 慢时间轴
% 固定地物杂波（多普勒频率为0）
clutter = 10^(CNR/20) * ones(1, N_pulse);  % 直流杂波
% 运动目标回波（复正弦信号）
target = exp(1j*2*pi*f_d*t_slow);  
% 加性高斯白噪声
noise = (randn(1, N_pulse) + 1j*randn(1, N_pulse))/sqrt(2) * 10^(-SNR/20);
% 总回波 = 杂波 + 目标 + 噪声
s_total = clutter + target + noise;
%% ===================== 3. 固定对消实现（一次/二次） =====================
% 固定一次对消（H(z)=1-z^-1 → y(n)=x(n)-x(n-1)）
s_1canc = s_total(2:end) - s_total(1:end-1);  % 向量化计算，避免循环
% 固定二次对消（H(z)=(1-z^-1)^2 → y(n)=x(n)-2x(n-1)+x(n-2)）
s_2canc = s_total(3:end) - 2*s_total(2:end-1) + s_total(1:end-2);
%% ===================== 4. 频率响应分析（验证对消器特性） =====================
N_fft = 1024;  % FFT点数
f = linspace(-PRF/2, PRF/2, N_fft);  % 多普勒频率轴
% 一次对消器频率响应
h1 = [1, -1];  % 一次对消器系数
H1 = fftshift(fft(h1, N_fft));
mag_H1 = 20*log10(abs(H1));
% 二次对消器频率响应
h2 = [1, -2, 1];  % 二次对消器系数
H2 = fftshift(fft(h2, N_fft));
mag_H2 = 20*log10(abs(H2));
%% ===================== 5. 结果可视化 =====================
figure(1);
% 子图1：原始回波（时域）
subplot(2,2,1);
plot(t_slow, real(s_total), 'b', 'LineWidth',1.2);
hold on; grid on; box on;
xlabel('慢时间 (s)','FontSize',11);
ylabel('回波幅度（实部）','FontSize',11);
title('原始回波（杂波+目标+噪声）','FontSize',13);
ylim([min(real(s_total))-1, max(real(s_total))+1]);
% 子图2：固定对消器频率响应
subplot(2,2,2);
plot(f, mag_H1, 'r', 'LineWidth',1.5, 'DisplayName','一次对消器');
hold on; grid on; box on;
plot(f, mag_H2, 'g', 'LineWidth',1.5, 'DisplayName','二次对消器');
xlabel('多普勒频率 (Hz)','FontSize',11);
ylabel('幅度响应 (dB)','FontSize',11);
title('固定对消器频率响应','FontSize',13);
xlim([-PRF/4, PRF/4]); ylim([-60, 10]);
legend('Location','best');
% 子图3：一次对消结果（时域）
subplot(2,2,3);
t_1canc = t_slow(2:end);
plot(t_1canc, real(s_1canc), 'r', 'LineWidth',1.2);
hold on; grid on; box on;
xlabel('慢时间 (s)','FontSize',11);
ylabel('对消后幅度（实部）','FontSize',11);
title('固定一次对消结果','FontSize',13);
% 子图4：二次对消结果（时域）
subplot(2,2,4);
t_2canc = t_slow(3:end);
plot(t_2canc, real(s_2canc), 'g', 'LineWidth',1.2);
hold on; grid on; box on;
xlabel('慢时间 (s)','FontSize',11);
ylabel('对消后幅度（实部）','FontSize',11);
title('固定二次对消结果','FontSize',13);
%% ===================== 6. 频域对比（杂波抑制效果） =====================
figure(2);
% 原始回波频谱
S_total = fftshift(fft(s_total, N_fft));
mag_total = 20*log10(abs(S_total));
plot(f, mag_total, 'b', 'LineWidth',1.2, 'DisplayName','原始回波');
% 一次对消后频谱
S_1canc = fftshift(fft(s_1canc, N_fft));
mag_1canc = 20*log10(abs(S_1canc));
hold on; plot(f, mag_1canc, 'r', 'LineWidth',1.2, 'DisplayName','一次对消后');
% 二次对消后频谱
S_2canc = fftshift(fft(s_2canc, N_fft));
mag_2canc = 20*log10(abs(S_2canc));
plot(f, mag_2canc, 'g', 'LineWidth',1.2, 'DisplayName','二次对消后');
grid on; box on;
xlabel('多普勒频率 (Hz)','FontSize',12);
ylabel('幅度 (dB)','FontSize',12);
title('固定对消前后的多普勒频谱','FontSize',14);
xlim([-PRF/4, PRF/4]);
legend('Location','best');




















