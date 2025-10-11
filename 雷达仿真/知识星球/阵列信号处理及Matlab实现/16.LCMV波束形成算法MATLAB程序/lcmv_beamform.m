% LCMV �����γɵ�MATLAB�������
% Developed by xiaofei zhang (�Ͼ����պ����ѧ ���ӹ���ϵ ��С�ɣ�
% EMAIL:zhangxiaofei@nuaa.edu.cn, fei_zxf@163.com

clc; 
close all
clear all; 
M=18;                                     %% the number of antennas
L=100;                                    %% sample number 
thetas=10;                                %�ź�����Ƕ� 
thetai=[-30 30];                          %��������Ƕ� 
n=[0:M-1]';%n 

vs=exp(-j*pi*n*sin(thetas/180*pi));       %�źŷ���ʸ�� 
vi=exp(-j*pi*n*sin(thetai/180*pi));       %���ŷ���ʸ�� 
f=16000;                                  % carrier frequency
t=[0:1:L-1]/200; 
snr=10;                                   %����� 
inr=10;                                   %����� 
 
xs=sqrt(10^(snr/10))*vs*exp(j*2*pi*f*t);  %���������ź� 
xi=sqrt(10^(inr/10)/2)*vi*[randn(length(thetai),L)+j*randn(length(thetai),L)];%��������ź�
noise=[randn(M,L)+j*randn(M,L)]/sqrt(2); %% noise
%% 
X=xi+noise;                              %% noisly received signal
R=X*X'/L;                                %% construct covariance matrix(X':X�Ĺ���ת��)
wop1=inv(R)*vs/(vs'*inv(R)*vs);          %% beamforming�������γɣ�(inv:��������)
sita=48*[-1:0.001:1];                    %% ɨ�跽��Χ
v=exp(-j*pi*n*sin(sita/180*pi));         %% ɨ�跽��ʸ�� 
B=abs(wop1'*v); 
plot(sita,20*log10(B/max(B)),'k'); 
title('����ͼ');xlabel('�Ƕ�/degree');ylabel('����ͼ/dB'); 
grid on 
axis([-48 48 -50 0]); 
hold off

