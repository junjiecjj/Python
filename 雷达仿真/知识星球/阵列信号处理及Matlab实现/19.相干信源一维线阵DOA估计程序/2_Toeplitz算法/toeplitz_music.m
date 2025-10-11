clear all;
close all;
clc;
source_number=2;%��Ԫ��
sensor_number=8;%��Ԫ��
N_x=1024; %�źų���
snapshot_number=N_x;%������
w=[107e6 107e6].';%�ź�Ƶ��
fs = 400e6;
l=(3e8/w(1)+3e8/w(2))/2;%�źŲ���
d=0.5*l;%��Ԫ���
snr=0;%�����

source_doa=[45 55];%�����źŵ�����Ƕ�
A=[exp(-1i*(0:sensor_number-1)*d*2*pi*sin(source_doa(1)*pi/180)/l);exp(-1i*(0:sensor_number-1)*d*2*pi*sin(source_doa(2)*pi/180)/l)].';%��������
s=10.^(snr/20)*exp(1i*2*pi*w*(0:N_x-1)/fs);%�����ź�
%rou = abs(mean(s(1,:).*conj(s(2,:)))/sqrt((mean(abs(s(1,:)).^2))*(mean(abs(s(2,:)).^2))));

%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+1i*randn(sensor_number,N_x));%���˸�˹������������н����ź�

R=x*x'/snapshot_number;
%������Э�������б�Խ����ϵ�Ԫ�ؽ���ƽ��,��Toeplitz��
dd=zeros(2*sensor_number-1);%15��15
for i=-(sensor_number-1):(sensor_number-1)
    c=sum(diag(R,i))/(sensor_number-abs(i));%ÿһ�Խ���ȡƽ��
    dd(i+sensor_number)=c; 
end

for k=1:sensor_number
      R(k,k)=dd(sensor_number);
end

for k=1:(sensor_number-1)
      R(k+1,k)=dd(sensor_number-1);
      R(k,k+1)=dd(sensor_number+1);
end

for k=1:(sensor_number-2)
      R(k+2,k)=dd(sensor_number-2);
      R(k,k+2)=dd(sensor_number+2);
end

for k=1:(sensor_number-3)
      R(k+3,k)=dd(sensor_number-3);
      R(k,k+3)=dd(sensor_number+3);
end

for k=1:(sensor_number-4)
      R(k+4,k)=dd(sensor_number-4);
      R(k,k+4)=dd(sensor_number+4);
end

for k=1:(sensor_number-5)
      R(k+5,k)=dd(sensor_number-5);
      R(k,k+5)=dd(sensor_number+5);
end

for k=1:(sensor_number-6)
      R(k+6,k)=dd(sensor_number-6);
      R(k,k+6)=dd(sensor_number+6);
end

for k=1:(sensor_number-7)
      R(k+7,k)=dd(sensor_number-7);
      R(k,k+7)=dd(sensor_number+7);
end

disp('R');
disp(R);
%��Toeplitz�����Э���������������ֽ⣬�õ��ź��ӿռ�������ӿռ�
[V,D]=eig(R);
D=diag(D);
disp(D);
Un=V(:,1:sensor_number-source_number);
Gn=Un*Un';
disp('Gn');
disp(Gn);
 
% [U,S,V]=svd(R);
% Un=U(:,source_number+1:sensor_number);
% Gn=Un*Un';

%����MUSIC�㷨�����׷��������Ӷ��õ������﷽��
% searching_doa=-90:0.1:90;%�����������ΧΪ-90~90��
searching_doa=-90:0.1:90;%�����������ΧΪ-90~90��
 for kk=1:length(searching_doa)
   a_theta=exp(-1i*(0:sensor_number-1)'*2*pi*d*sin(pi*searching_doa(kk)/180)/l);
   Pmusic(kk)=1./abs((a_theta)'*Gn*a_theta);
 end
plot(searching_doa,10*log10(Pmusic));
%axis([-90 90 -90 90]);
xlabel('�����/��');
ylabel('�׷�/dB');
% legend('TOEPLITZ MUSIC SPECTRUM');
title('Toeplitz����MUSIC����');
grid on;
