


% 3 | 数据滤波：探讨卡尔曼滤波、SG滤波与组合滤波
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247486011&idx=1&sn=cd136895f56c14559a78d897819dcb95&chksm=c15b9b15f62c1203d59a010088c303a57424cf30f1dda50e20dd18a2a3ab0b0dcc65f30dd037&cur_album_id=3587607448191893505&scene=190#rd


clc
clear
close all

Fs = 1000;
t = 0:1/Fs:1;
signal = sin(2 * pi * 50 * t) + 0.5 * randn(size(t));
window_lengths = [5, 11, 21, 31];
poly_orders = [2, 3, 4];

figure;
for i = 1:length(window_lengths)
    for j = 1:length(poly_orders)

        window_len = window_lengths(i);
        poly_order = poly_orders(j);
        filtered_signal = sgolayfilt(signal, poly_order, window_len);
        subplot(length(window_lengths), length(poly_orders), (i-1)*length(poly_orders) + j);
        plot(t, signal, 'b', 'DisplayName', '原始信号');
        hold on;
        plot(t, filtered_signal, 'r', 'DisplayName', '滤波后信号');
        hold off;

        title(['窗口长度 = ' num2str(window_len) ', 多项式阶数 = ' num2str(poly_order)]);
        xlabel('时间 (秒)');
        ylabel('幅值');
        legend show;
        grid on;
    end
end
