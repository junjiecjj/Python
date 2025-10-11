%VECTOR_SINGULAR_VALUE_MUSIC ALOGRITHM
%DOA ESTIMATION BY VECTOR_SINGULAR_VALUE_MUSIC
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
snr=10;%�����

source_doa=[45 -60];%�����źŵ�����Ƕ�
A=[exp(-j*(0:sensor_number-1)*d*2*pi*sin(source_doa(1)*pi/180)/l);exp(-j*(0:sensor_number-1)*d*2*pi*sin(source_doa(2)*pi/180)/l)].';%��������
s=10.^(snr/20)*exp(j*w*[0:N_x-1]);%�����ź�

%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+j*randn(sensor_number,N_x));%���˸�˹������������н����ź�

R=x*x'/snapshot_number;
[Ur,Sr,Vr]=svd(R);%����Э���������������ֽ�
disp(Ur);
E=Ur(:,1);%��ȡ�����������
disp(E);
%���о����ع�
Y=[E(1,1) E(2,1) E(3,1) E(4,1) E(5,1) E(6,1);E(2,1) E(3,1) E(4,1) E(5,1) E(6,1) E(7,1);E(3,1) E(4,1) E(5,1) E(6,1) E(7,1) E(8,1)].'
disp(Y);

%���ع��ľ����������ֵ�ֽ�
%[U,S,V]=svd(Y);
%Un=V(:,1:m-source_number);
%Gn=Un*Un';
[U,S,V]=svd(Y);
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
legend('VECTOR-SINGULAR-VALUE-MUSIC SPECTRUM');
title('ʸ������ֵMUSIC����');
grid on;