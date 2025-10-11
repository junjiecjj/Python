%% 参数设置
clear all;clc ;close all 

fs=100e6;%100Mhz
fc=10e6;%10Mhz
T=10.24e-6;
B=40e6;
N=T*fs;
t=-T/2:1/fs:T/2-1/fs;
f=(-N/2:N/2-1)*B/N;
%% LFM信号
K=B/T;
s0=exp(j*2*pi*fc+j*pi*K*t.^2);
% figure(1)
% plot(t,s0);
% [afmag0,delay0,doppler0]=ambgfun(s0,fs,fs/10);
% figure(2)
% mesh(delay0,doppler0,afmag0);
% xlim([-1e-5,1e-5]);
% ylim([-4e7,4e7]);
% afmag0=10*log(afmag0);
% figure;
% plot(delay0,afmag0(1025,:),'-o');
% xlim([-0.5e-6,0.5e-6]);
% ylim([-70,0]);
% figure;
% plot(doppler0,afmag0(:,1020),'-o');
% xlim([-0.4e6,0.4e6]);
% ylim([-70,0]);

%% MSK调制
% 参数设置
m = 2;          %进制数
L = 2;          %关联长度，记忆长度
h_m = 1;h_p = 2;
h = h_m/h_p;    %调制指数
sps = 32;        %每个符号样点数 sample per symbol
[g,q] = rc_pulse(sps,L);%生成升余弦脉冲函数g，及其积分函数q
sum_bit=32;
signal_bit = [ 1 0 1 0 1 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1 0 1 0 1 0 1 0 1 1 0 1];
% signal_bit = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
temp = reshape(signal_bit,log2(m),sum_bit/log2(m));%串并转换 
temp = temp';
temp = bi2de(temp,'left-msb');
symbol = (2*temp-m+1)';%码元符号,行向量        
sMSK= cpm_mod(symbol,h,sps,L,q,m);
%% MSK_LFM信号
sMSK=s0.*sMSK;
%% OFDM调制
sMSK_ofdm=reshape(sMSK,2,[]);
IFFT_sMSK_ofdm=ifft(sMSK_ofdm);
tx_MSK_ofdm=reshape(IFFT_sMSK_ofdm,1,[]);
%% 画图
[afmag02,delay02,doppler02]=ambgfun(tx_MSK_ofdm,fs,fs/10);
figure(5)
% contour(delay,doppler,afmag,'ShowText','on');
mesh(delay02,doppler02,afmag02);
xlim([-1e-5,1e-5]);
ylim([-4e7,4e7]);
% ambgu=afmag.*(afmag>0.5);
ylabel('时延')
xlabel('多普勒频移')
zlabel('幅度')
title('速度分辨率')
afmag02=10*log(afmag02);

figure(3);
a=delay02/max(delay02)
plot(a,afmag02(1025,:),'-o');

xlim([-1,1]);
% xlim([-0.5e-6,0.5e-6]);
ylim([-70,0]);
xlabel('归一化时延')
ylabel('幅度/dB')
title('距离分辨率')
figure(4);
plot(doppler02/max(doppler02),afmag02(:,1020),'-o');
xlim([-1,1]);
% xlim([-0.4e6,0.4e6]);
ylim([-70,0]);
xlabel('归一化多普勒频移')
ylabel('幅度/dB')
title('速度分辨率')
