%MATRIX_DECOMPOSE__MUSIC ALOGRITHM,MD�㷨
%DOA ESTIMATION BY MATRIX_DECOMPOSE__MUSIC
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
m=4;%ÿ��������Ԫ��
% m=6;%������Լ������,Ҳ���Բ��
l0=3;%l0<(M-m)
snr=0;%�����

source_doa=[45 55];%�����źŵ�����Ƕ�
A=[exp(-j*(0:sensor_number-1)*d*2*pi*sin(source_doa(1)*pi/180)/l);exp(-j*(0:sensor_number-1)*d*2*pi*sin(source_doa(2)*pi/180)/l)].';%��������
s=10.^(snr/20)*exp(j*w*[0:N_x-1]);%�����ź�

%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+j*randn(sensor_number,N_x));%���˸�˹������������н����ź�

R=x*x'/snapshot_number;
%���о����ع�:
R0=R([1:4],:);
R1=R([2:5],:);
R2=R([3:6],:);
R3=R([4:7],:);
Rm=[R0 R1 R2 R3]; 
%���о����ع�:
% R0=R([1:6],:);
% R1=R([2:7],:);
% R2=R([3:8],:);
% Rm=[R0 R1 R2]; 
% ���ع��ľ����������ֵ�ֽ�
[U,S,V]=svd(Rm);
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
legend('MATRIX-DECOMPOSE-MUSIC SPECTRUM');
title('����ֽ�MUSIC����');
grid on;