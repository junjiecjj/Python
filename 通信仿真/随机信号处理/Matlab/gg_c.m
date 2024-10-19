


% 7 | 随机信号分析与应用：从自相关到功率谱密度的探讨（上）
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485857&idx=1&sn=536464feabfb9d80030cb3a67a963441&chksm=c15b988ff62c119947b6616a2315bf78aff223fd57a7697ff9bdc227f8278c513de479dddeb9&cur_album_id=3587607448191893505&scene=190#rd


clc
clear
close all

clc
clear
close all

fs = 1000;
t = 0:1/fs:1-1/fs;
f1 = 50;
f2 = 120;
x = sin(2*pi*f1*t) + sin(2*pi*f2*t) + 0.5*randn(size(t)); % 加入噪声的信号
x = x - mean(x);

N1 = 256;
N2 = 512;
[Pxx1, f1] = periodogram(x(1:N1), [], N1, fs);
[Pxx2, f2] = periodogram(x(1:N2), [], N2, fs);

window_hann = hanning(N2);%haning
window_hamm = hamming(N2);%haming
[Pxx_hann, f_hann] = periodogram(x(1:N2), window_hann, N2, fs);
[Pxx_hamm, f_hamm] = periodogram(x(1:N2), window_hamm, N2, fs);
%%
figure;
subplot(2,1,1);
plot(f1,10*log10(Pxx1),'b', 'LineWidth', 1.5); hold on;
plot(f2,10*log10(Pxx2),'r--', 'LineWidth', 1.5);
xlabel('频率 (Hz)', 'FontSize', 12);
ylabel('功率谱 (dB)', 'FontSize', 12);
title('不同信号长度下的功率谱', 'FontSize', 14);
legend(['N = ', num2str(N1)], ['N = ', num2str(N2)]);
grid on;

subplot(2,1,2);
plot(f_hann,10*log10(Pxx_hann),'g', 'LineWidth', 1.5); hold on;
plot(f_hamm,10*log10(Pxx_hamm),'m--', 'LineWidth', 1.5);
xlabel('频率 (Hz)', 'FontSize', 12);
ylabel('功率谱 (dB)', 'FontSize', 12);
title('不同窗口函数下的功率谱', 'FontSize', 14);
legend('Hanning窗', 'Hamming窗');
grid on;
