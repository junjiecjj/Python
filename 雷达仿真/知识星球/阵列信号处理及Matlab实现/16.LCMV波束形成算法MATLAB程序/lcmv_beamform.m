% LCMV 波束形成的MATLAB仿真程序
% Developed by xiaofei zhang (南京航空航天大学 电子工程系 张小飞）
% EMAIL:zhangxiaofei@nuaa.edu.cn, fei_zxf@163.com

clc; 
close all
clear all; 
M=18;                                     %% the number of antennas
L=100;                                    %% sample number 
thetas=10;                                %信号入射角度 
thetai=[-30 30];                          %干扰入射角度 
n=[0:M-1]';%n 

vs=exp(-j*pi*n*sin(thetas/180*pi));       %信号方向矢量 
vi=exp(-j*pi*n*sin(thetai/180*pi));       %干扰方向矢量 
f=16000;                                  % carrier frequency
t=[0:1:L-1]/200; 
snr=10;                                   %信噪比 
inr=10;                                   %干噪比 
 
xs=sqrt(10^(snr/10))*vs*exp(j*2*pi*f*t);  %构造有用信号 
xi=sqrt(10^(inr/10)/2)*vi*[randn(length(thetai),L)+j*randn(length(thetai),L)];%构造干扰信号
noise=[randn(M,L)+j*randn(M,L)]/sqrt(2); %% noise
%% 
X=xi+noise;                              %% noisly received signal
R=X*X'/L;                                %% construct covariance matrix(X':X的共轭转置)
wop1=inv(R)*vs/(vs'*inv(R)*vs);          %% beamforming（波束形成）(inv:矩阵求逆)
sita=48*[-1:0.001:1];                    %% 扫描方向范围
v=exp(-j*pi*n*sin(sita/180*pi));         %% 扫描方向矢量 
B=abs(wop1'*v); 
plot(sita,20*log10(B/max(B)),'k'); 
title('波束图');xlabel('角度/degree');ylabel('波束图/dB'); 
grid on 
axis([-48 48 -50 0]); 
hold off

