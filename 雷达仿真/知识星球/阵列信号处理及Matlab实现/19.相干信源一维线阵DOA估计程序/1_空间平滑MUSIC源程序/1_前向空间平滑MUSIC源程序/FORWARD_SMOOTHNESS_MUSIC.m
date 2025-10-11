%FORWARD_SMOOTHNESS_MUSIC ALOGRITHM
%DOA ESTIMATION BY FORWARD_SMOOTHNESS_MUSIC
clear all;
close all;
clc;

source_number=2;%信元数
sensor_number=8;%阵元数
N_x=1024; %信号长度
snapshot_number=N_x;%快拍数
w=[pi/4 pi/4].';%信号频率
l=((2*pi*3e8)/w(1)+(2*pi*3e8)/w(2))/2;%信号波长  
d=0.5*l;%阵元间距
m=6;%每个子阵阵元数
p=3;%相互交错的子阵数
snr=0;%信噪比

theta1=45;theta2=-60;%两个信号的入射角度
A=[exp(-j*(0:sensor_number-1)*d*2*pi*sin(theta1*pi/180)/l);exp(-j*(0:sensor_number-1)*d*2*pi*sin(theta2*pi/180)/l)].';%阵列流型

s=sqrt(10.^(snr/10))*exp(j*w*[0:N_x-1]);%仿真信号

%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+j*randn(sensor_number,N_x));%加了高斯白噪声后的阵列接收信号

x1=x([1:6],:);R1=x1*x1'/1024;
x2=x([2:7],:);R2=x2*x2'/1024;
x3=x([3:8],:);R3=x3*x3'/1024;

Rf=(R1+R2+R3)/3;

% [V,D]=eig(Rf);
% Un=V(:,1:m-source_number);
% Gn=Un*Un';
[U,S,V]=svd(Rf);
Un=U(:,source_number+1:m);
Gn=Un*Un';

searching_doa=-90:0.1:90;%线阵的搜索范围为-90~90度
 for i=1:length(searching_doa)
   a_theta=exp(-j*(0:m-1)'*2*pi*d*sin(pi*searching_doa(i)/180)/l);
   Pmusic(i)=1./abs((a_theta)'*Gn*a_theta);
 end

plot(searching_doa,10*log(Pmusic));
%axis([-90 90 -90 90]);
xlabel('入射角/度');
ylabel('谱峰、dB');
legend('FORWARD-SMOOTHNESS-MUSIC SPECTRUM');
title('前向空间平滑MUSIC估计');
grid on;