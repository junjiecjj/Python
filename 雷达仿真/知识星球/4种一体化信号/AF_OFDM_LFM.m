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


%% OFDM调制
sQAM_ofdm=reshape(s0,2,[]);
IFFT_sQAM_ofdm=ifft(sQAM_ofdm);
tx_QAM_ofdm=reshape(IFFT_sQAM_ofdm,1,[]);
%% 画图QAM-LFM 模糊函数
% figure;
% plot(t,sQAM);
% [afmag01,delay01,doppler01]=ambgfun(sQAM,fs,fs/10);
% figure(4)
% mesh(delay01,doppler01,afmag01);
% xlim([-1e-5,1e-5]);
% ylim([-4e7,4e7]);
%% 画图OFDM-QAM-LFM 模糊函数
% figure;
% plot(t,tx_QAM_ofdm);
[afmag02,delay02,doppler02]=ambgfun(tx_QAM_ofdm,fs,fs/10);
figure(5)
% contour(delay,doppler,afmag,'ShowText','on');
mesh(delay02,doppler02,afmag02);
xlim([-1e-5,1e-5]);
ylim([-4e7,4e7]);
% ambgu=afmag.*(afmag>0.5);


