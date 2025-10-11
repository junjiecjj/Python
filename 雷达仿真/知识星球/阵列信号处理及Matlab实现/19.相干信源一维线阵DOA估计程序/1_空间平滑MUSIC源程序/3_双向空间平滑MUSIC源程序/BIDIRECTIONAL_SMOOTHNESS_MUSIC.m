%BIDIRECTIONAL_SMOOTHNESS_MUSIC ALOGRITHM
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
snr=10;%�����
theta=[45 25];%�����źŵ�����Ƕ�
A=[exp(-j*(0:sensor_number-1)*d*2*pi*sin(theta(1)*pi/180)/l);exp(-j*(0:sensor_number-1)*d*2*pi*sin(theta(2)*pi/180)/l)].';%��������

s=sqrt(10.^(snr/10))*exp(j*w*[0:N_x-1]);%�����ź�
%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+j*randn(sensor_number,N_x));%���˸�˹������������н����ź�
%ǰ��ƽ��
xf1=x([1:6],:);Rf1=xf1*xf1'/snapshot_number;
xf2=x([2:7],:);Rf2=xf2*xf2'/snapshot_number;
xf3=x([3:8],:);Rf3=xf3*xf3'/snapshot_number;
Rf=(Rf1+Rf2+Rf3)/3;
%����ƽ��
xb1=conj(x([8:-1:3],:));Rb1=xb1*xb1'/snapshot_number;
xb2=conj(x([7:-1:2],:));Rb2=xb2*xb2'/snapshot_number;
xb3=conj(x([6:-1:1],:));Rb3=xb3*xb3'/snapshot_number;
Rb=(Rb1+Rb2+Rb3)/3;
%˫��ƽ��
Rbf=(Rf+Rb)/2;

%[V,D]=eig(Rbf);
%Un=V(:,1:m-source_number);
%Gn=Un*Un';
[U,S,V]=svd(Rbf);
Un=U(:,source_number+1:m);
Gn=Un*Un';

searching_doa=-90:0.1:90;%�����������ΧΪ-90~90��
 for i=1:length(searching_doa)
   a_theta=exp(-j*(0:m-1)'*2*pi*d*sin(pi*searching_doa(i)/180)/l);
   Pmusic(i)=1./abs((a_theta)'*Gn*a_theta);
 end
plot(searching_doa,10*log(Pmusic),'r');

[value maxindex] = findpeaks(Pmusic);
Pmin = min(Pmusic);
[~,index] = max(value);
thetaEst(1,1) = searching_doa(maxindex(index));
value(index) = Pmin;
[~,index] = max(value);
thetaEst(1,2) = searching_doa(maxindex(index));
%axis([-90 90 -90 90]);
xlabel('�����/degree');
ylabel('�׷�/dB');
legend('BIDIRECTIONAL-SMOOTHNESS-MUSIC SPECTRUM');
title('˫��ռ�ƽ��MUSIC����');
grid on;