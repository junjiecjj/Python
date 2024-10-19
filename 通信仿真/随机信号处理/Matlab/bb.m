% 2 | 随机信号分析与应用：从自相关到功率谱密度的探讨
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485696&idx=1&sn=fcee54640e6a9c43d4bc55b3f93747fc&chksm=c15b982ef62c1138c469c3dba681b595af307f7171388b818729ed85bfdb01578be722764e40&cur_album_id=3587607448191893505&scene=190#rd


close all;
clear all;
clc;

% 基本参数
fs = 1e6;                 % 采样频率
t = 0:1/fs:1;             % 时间向量
c = 3e8;                  % 光速，假设信号以光速传播
d = 450;                  % 目标实际距离（米）
delay = 2*d/c;            % 实际延迟时间（秒）

X = chirp(t, 0, 1, 100);
Y = [zeros(1, round(delay*fs)), X(1:end-round(delay*fs))] + 0.5*randn(size(X));
[R, lag] = xcorr(Y, X);
[~, I] = max(R);
time_delay = lag(I)/fs;
estimated_distance = time_delay * c / 2;

fprintf('实际的目标距离为 %.2f 米\n', d);
fprintf('估计的目标距离为 %.2f 米\n', estimated_distance);

figure;
subplot(3, 1, 1);
plot(t, X);
title('发射信号 X(t)');
xlabel('时间 (秒)');
ylabel('幅度');

subplot(3, 1, 2);
plot(t, Y(1:length(t)));
title('接收信号 Y(t)');
xlabel('时间 (秒)');
ylabel('幅度');

subplot(3, 1, 3);
plot(lag/fs, R);
title('互相关函数 R_{XY}(\tau)');
xlabel('时间延迟 (秒)');
ylabel('互相关值');

figure;
plot([0, d], [0, 0], 'bo-', 'LineWidth', 2, 'MarkerSize', 10);
hold on;
text(0, 10, '发射站', 'HorizontalAlignment', 'center');
text(d, 10, sprintf('目标 (%.2f 米)', d), 'HorizontalAlignment', 'center');
plot(d, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
line([0, d], [0, 0], 'Color', 'k', 'LineStyle', '--');
plot(estimated_distance, 0, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
text(estimated_distance, -10, sprintf('预测位置 (%.2f 米)', estimated_distance), 'HorizontalAlignment', 'center');
xlabel('距离 (米)');
ylabel('高度');
title('目标与发射站示意图');
xlim([-50, max(d, estimated_distance) + 50]);
ylim([-50, 50]);
axis equal;
grid on;

