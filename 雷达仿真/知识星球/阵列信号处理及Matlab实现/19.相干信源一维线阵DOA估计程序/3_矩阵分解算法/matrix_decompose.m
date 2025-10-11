%MATRIX_DECOMPOSE__MUSIC ALOGRITHM,MD算法
%DOA ESTIMATION BY MATRIX_DECOMPOSE__MUSIC
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
m=4;%每个子阵阵元数
% m=6;%不满足约束条件,也可以测角
l0=3;%l0<(M-m)
snr=0;%信噪比

source_doa=[45 55];%两个信号的入射角度
A=[exp(-j*(0:sensor_number-1)*d*2*pi*sin(source_doa(1)*pi/180)/l);exp(-j*(0:sensor_number-1)*d*2*pi*sin(source_doa(2)*pi/180)/l)].';%阵列流型
s=10.^(snr/20)*exp(j*w*[0:N_x-1]);%仿真信号

%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+j*randn(sensor_number,N_x));%加了高斯白噪声后的阵列接收信号

R=x*x'/snapshot_number;
%进行矩阵重构:
R0=R([1:4],:);
R1=R([2:5],:);
R2=R([3:6],:);
R3=R([4:7],:);
Rm=[R0 R1 R2 R3]; 
%进行矩阵重构:
% R0=R([1:6],:);
% R1=R([2:7],:);
% R2=R([3:8],:);
% Rm=[R0 R1 R2]; 
% 对重构的矩阵进行奇异值分解
[U,S,V]=svd(Rm);
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
legend('MATRIX-DECOMPOSE-MUSIC SPECTRUM');
title('矩阵分解MUSIC估计');
grid on;