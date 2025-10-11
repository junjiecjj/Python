%VECTOR_SINGULAR_VALUE_MUSIC ALOGRITHM
%DOA ESTIMATION BY VECTOR_SINGULAR_VALUE_MUSIC
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
m=5;%每个子阵阵元数
p=3;%相互交错的子阵数
snr=30;%信噪比

source_doa=[45 -60];%两个信号的入射角度
A=[exp(-j*(0:sensor_number-1)*d*2*pi*sin(source_doa(1)*pi/180)/l);exp(-j*(0:sensor_number-1)*d*2*pi*sin(source_doa(2)*pi/180)/l)].';%阵列流型
s=10.^(snr/20)*exp(j*w*[0:N_x-1]);%仿真信号
%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+j*randn(sensor_number,N_x));%加了高斯白噪声后的阵列接收信号8*1024
x2=x(1,:);%参考阵元矢量
y1=x*x2'/snapshot_number;%8*1

%进行矩阵重构
% Y=[y1(1,1) y1(2,1) y1(3,1) y1(4,1) y1(5,1) y1(6,1);y1(2,1) y1(3,1) y1(4,1) y1(5,1) y1(6,1) y1(7,1);y1(3,1) y1(4,1) y1(5,1) y1(6,1) y1(7,1) y1(8,1)].'
Y=[y1(1,1) y1(2,1) y1(3,1) y1(4,1) y1(5,1) ;y1(2,1) y1(3,1) y1(4,1) y1(5,1) y1(6,1) ;y1(3,1) y1(4,1) y1(5,1) y1(6,1) y1(7,1) ; y1(4,1) y1(5,1) y1(6,1) y1(7,1) y1(8,1)].'

disp(Y);%6*3

%[U,S,V]=svd(Y);
%Un=V(:,1:m-source_number);
%Gn=Un*Un';
[U,S,V]=svd(Y);
Un=U(:,source_number+1:m);  %6*4
Gn=Un*Un';%6*6

searching_doa=-90:0.1:90;%线阵的搜索范围为-90~90度
 for i=1:length(searching_doa)
   a_theta=exp(-1i*(0:m-1)'*2*pi*d*sin(pi*searching_doa(i)/180)/l);%6*1
   Pmusic(i)=1./abs((a_theta)'*Gn*a_theta);
 end

plot(searching_doa,10*log(Pmusic));
%axis([-90 90 -90 90]);
xlabel('入射角/度');
ylabel('谱峰、dB');
legend('VECTOR-SINGULAR-VALUE-MUSIC SPECTRUM');
title('DSVDMUSIC估计');
grid on;
