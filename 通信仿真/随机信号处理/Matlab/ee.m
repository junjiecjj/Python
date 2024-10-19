% 4 | 随机信号分析与应用：从自相关到功率谱密度的探讨
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485778&idx=1&sn=09ec65d7f65cc25cb944f0b895b6a05b&chksm=c15b987cf62c116a7cccf99e57ed8314ca03f77b6763b48a1152eae5aeeb7dd8e1583c049a0c&cur_album_id=3587607448191893505&scene=190#rd


% 3.Parseval定理


clc
clear
close all

fs = 1000;                % 采样频率
t = 0:1/fs:1-1/fs;        % 时间向量
f1 = 50;                  % 信号频率1
f2 = 120;                 % 信号频率2
x = cos(2*pi*f1*t) + 0.5*cos(2*pi*f2*t) + randn(size(t));  % 生成信号（包含噪声）

% 计算功率谱密度
[Sx, f] = periodogram(x, [], [], fs);

%%
figure;

% 非负性
subplot(3,1,1);
plot(f, Sx, 'LineWidth', 1.5);
xlabel('频率 (Hz)', 'FontSize', 12);
ylabel('功率谱密度', 'FontSize', 12);
title('功率谱密度的非负性', 'FontSize', 14);
grid on;

% 对称性
subplot(3,1,2);
plot(f, Sx, 'LineWidth', 1.5);
hold on;
plot(-f, Sx, '--', 'LineWidth', 1.5);
xlabel('频率 (Hz)', 'FontSize', 12);
ylabel('功率谱密度', 'FontSize', 12);
title('功率谱密度的对称性', 'FontSize', 14);
legend('S_X(f)', 'S_X(-f)', 'FontSize', 12);
grid on;

% Parseval定理验证
total_power_time_domain = mean(x.^2);
total_power_freq_domain = sum(Sx) * (f(2) - f(1));

subplot(3,1,3);
bar(1, total_power_time_domain, 'FaceColor', 'b');
hold on;
bar(2, total_power_freq_domain, 'FaceColor', 'r');
set(gca, 'XTickLabel', {'时域总功率', '频域总功率'}, 'FontSize', 12);
ylabel('功率', 'FontSize', 12);
title('Parseval定理验证', 'FontSize', 14);
legend('时域', '频域', 'FontSize', 12);
grid on;
sgtitle('功率谱密度的基本性质验证', 'FontSize', 16);


% 5.2 功率谱密度的实际意义

clc
clear
close all

fs = 1000;                    % 采样频率
t = 0:1/fs:1-1/fs;            % 时间向量
f1 = 50;                      % 信号频率1
f2 = 150;                     % 信号频率2
signal = cos(2*pi*f1*t) + 0.5*cos(2*pi*f2*t);  % 生成两个正弦波信号
noise = 0.3*randn(size(t));    % 生成高斯白噪声
x = signal + noise;            % 信号与噪声相加

figure;
subplot(3,1,1);
plot(t, x, 'LineWidth', 1.5);
xlabel('时间 (秒)', 'FontSize', 12);
ylabel('幅度', 'FontSize', 12);
title('含噪声的复合信号', 'FontSize', 14);
grid on;

%%
[Sx, f] = periodogram(x, [], [], fs);
subplot(3,1,2);
plot(f, Sx, 'LineWidth', 1.5);
xlabel('频率 (Hz)', 'FontSize', 12);
ylabel('功率谱密度', 'FontSize', 12);
title('信号的功率谱密度', 'FontSize', 14);
grid on;

%%
d = designfilt('bandpassiir', 'FilterOrder', 8, ...
               'HalfPowerFrequency1', 45, 'HalfPowerFrequency2', 55, ...
               'SampleRate', fs);
filtered_signal = filtfilt(d, x);
subplot(3,1,3)
[Sx_filtered, f_filtered] = periodogram(filtered_signal, [], [], fs);
plot(f_filtered, Sx_filtered, 'LineWidth', 1.5);
xlabel('频率 (Hz)', 'FontSize', 12);
ylabel('功率谱密度', 'FontSize', 12);
title('滤波后信号的功率谱密度', 'FontSize', 14);
grid on;



% 5.3 功率谱密度与信号特征的关系
% 1. 正弦波信号的功率谱密度分析
clc
clear
close all

fs = 1000;                    % 采样频率
t = 0:1/fs:1-1/fs;            % 时间向量
x = cos(2*pi*100*t);          % 100 Hz 的正弦波信号
[Pxx, f] = periodogram(x, [], [], fs);

figure;
plot(f, Pxx);
title('正弦波信号的功率谱密度');
xlabel('频率 (Hz)');
ylabel('功率/频率 (dB/Hz)');



% 2. 方波信号的功率谱密度分析
clc
claear
close all

fs = 1000;                    % 采样频率
t = 0:1/fs:1-1/fs;            % 时间向量
x = square(2*pi*100*t);       % 100 Hz 的方波信号
[Pxx, f] = periodogram(x, [], [], fs);

figure;
plot(f, Pxx);
title('方波信号的功率谱密度');
xlabel('频率 (Hz)');
ylabel('功率/频率 (dB/Hz)');


%3. 白噪声信号的功率谱密度分析
clc
clear
close all

fs = 1000;                    % 采样频率
t = 0:1/fs:1-1/fs;            % 时间向量
x = randn(1, fs);             % 白噪声信号
[Pxx, f] = periodogram(x, [], [], fs);

figure;
plot(f, Pxx);
title('白噪声信号的功率谱密度');
xlabel('频率 (Hz)');
ylabel('功率/频率 (dB/Hz)');

% 4. 短时脉冲信号的功率谱密度分析


clc
clear
close all
fs = 1000;                    % 采样频率
t = 0:1/fs:1-1/fs;            % 时间向量
x = [zeros(1, 450) ones(1, 100) zeros(1, 450)];  % 短时脉冲信号
[Pxx, f] = periodogram(x, [], [], fs);

figure;
plot(f, Pxx);
title('短时脉冲信号的功率谱密度');
xlabel('频率 (Hz)');
ylabel('功率/频率 (dB/Hz)');






















