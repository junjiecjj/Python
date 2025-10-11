%FORWARD_SMOOTHNESS_MUSIC ALOGRITHM
%DOA ESTIMATION BY FORWARD_SMOOTHNESS_MUSIC
clear all;
close all;
clc;

source_number=2;%��Ԫ��
sensor_number=8;%��Ԫ��
N_x=1024; %�źų���
snapshot_number=N_x;%������
w=[pi/4 pi/4].';%�ź�Ƶ��
l=((2*pi*3e8)/w(1)+(2*pi*3e8)/w(2))/2;%�źŲ���  
d=0.5*l;%��Ԫ���
m=6;%ÿ��������Ԫ��
p=3;%�໥�����������
snr=0;%�����

theta1=45;theta2=-60;%�����źŵ�����Ƕ�
A=[exp(-j*(0:sensor_number-1)*d*2*pi*sin(theta1*pi/180)/l);exp(-j*(0:sensor_number-1)*d*2*pi*sin(theta2*pi/180)/l)].';%��������

s=sqrt(10.^(snr/10))*exp(j*w*[0:N_x-1]);%�����ź�

%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+j*randn(sensor_number,N_x));%���˸�˹������������н����ź�

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

searching_doa=-90:0.1:90;%�����������ΧΪ-90~90��
 for i=1:length(searching_doa)
   a_theta=exp(-j*(0:m-1)'*2*pi*d*sin(pi*searching_doa(i)/180)/l);
   Pmusic(i)=1./abs((a_theta)'*Gn*a_theta);
 end

plot(searching_doa,10*log(Pmusic));
%axis([-90 90 -90 90]);
xlabel('�����/��');
ylabel('�׷塢dB');
legend('FORWARD-SMOOTHNESS-MUSIC SPECTRUM');
title('ǰ��ռ�ƽ��MUSIC����');
grid on;