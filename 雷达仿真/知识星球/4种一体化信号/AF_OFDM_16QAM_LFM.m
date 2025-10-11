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

%% 16QAM调制
nsymbol=1024;%表示一共有多少个符号，这里定义100000个符号
M=16;%M表示QAM调制的阶数,表示16QAM，16QAM采用格雷映射(所有星座点图均采用格雷映射)
graycode=[0 1 3 2 4 5 7 6 12 13 15 14 8 9 11 10];%格雷映射编码规则
msg=randi([0,M-1],1,nsymbol);%0到15之间随机产生一个数,数的个数为：1乘nsymbol，得到原始数据
msg1=graycode(msg+1);%对数据进行格雷映射
msgmod=qammod(msg1,M);%调用matlab中的qammod函数，16QAM调制方式的调用(输入0到15的数，M表示QAM调制的阶数)得到调制后符号
sQAM=msgmod;
sQAM=sQAM.*exp((j*2*pi*fc+j*pi*K*t.^2));
%% OFDM调制
sQAM_ofdm=reshape(sQAM,2,[]);
IFFT_sQAM_ofdm=ifft(sQAM_ofdm);
tx_QAM_ofdm=reshape(IFFT_sQAM_ofdm,1,[]);
[afmag02,delay02,doppler02]=ambgfun(tx_QAM_ofdm,fs,fs/10);
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


