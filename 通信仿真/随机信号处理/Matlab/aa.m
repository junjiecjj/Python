

close all;
clear all;
clc;


clc
clear
close all

f0 = 5;  
T = 1;  
Fs = 1000;  
t = 0:1/Fs:T;  
X_sin = sin(2*pi*f0*t);
X_rand = randn(size(t));
[acf_sin, lags_sin] = xcorr(X_sin, 'coeff');
[acf_rand, lags_rand] = xcorr(X_rand, 'coeff');

figure;
subplot(2,1,1);
plot(lags_sin/Fs, acf_sin);
title('正弦波信号的自相关函数');
xlabel('滞后时间 (秒)');
ylabel('自相关系数');
grid on;

subplot(2,1,2);
plot(lags_rand/Fs, acf_rand);
title('随机信号的自相关函数');
xlabel('滞后时间 (秒)');
ylabel('自相关系数');
grid on;