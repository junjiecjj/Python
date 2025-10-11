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

%% BPSK调制
%%基带信号产生
code = round(rand(1,1024));  % 二进制随机序列
%%BPSK基带调制
s = (code - 1/2) * 2;      % 双极性不归零序列
s =exp(j*s*fc*pi);
sBPSK=s.*s0;
%% OFDM调制
sBPSK_ofdm=reshape(sBPSK,2,[]);
IFFT_sBPSK_ofdm=ifft(sBPSK_ofdm);
tx_sBPSK_ofdm=reshape(IFFT_sBPSK_ofdm,1,[]);
%% 画图BPSK-LFM 模糊函数
% figure;
% plot(t,sBPSK);
% [afmag0,delay0,doppler0]=ambgfun(sBPSK,fs,fs/10);
% figure
% % contour(delay,doppler,afmag,'ShowText','on');
% mesh(delay0,doppler0,afmag0);
% xlim([-1e-5,1e-5]);
% ylim([-4e7,4e7]);
% % ambgu=afmag.*(afmag>0.5);
%% 画图OFDM-BPSK-LFM 模糊函数
% figure(3)
% plot(t,tx_sBPSK_ofdm);
[afmag02,delay02,doppler02]=ambgfun(tx_sBPSK_ofdm,fs,fs/10);
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
% ylim([-70,0]);
xlabel('归一化时延')
ylabel('幅度/dB')
title('距离分辨率')
figure(4);
plot(doppler02/max(doppler02),afmag02(:,1020),'-o');
xlim([-1,1]);
% xlim([-0.4e6,0.4e6]);
% ylim([-70,0]);
xlabel('归一化多普勒频移')
ylabel('幅度/dB')
title('速度分辨率')

