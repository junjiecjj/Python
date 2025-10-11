clear all;
close all;
clc;
source_number=2;%信元数
sensor_number=8;%阵元数
N_x=1024; %信号长度
snapshot_number=N_x;%快拍数
w=[107e6 107e6].';%信号频率
fs = 400e6;
l=(3e8/w(1)+3e8/w(2))/2;%信号波长
d=0.5*l;%阵元间距
snr=0;%信噪比

source_doa=[45 55];%两个信号的入射角度
A=[exp(-1i*(0:sensor_number-1)*d*2*pi*sin(source_doa(1)*pi/180)/l);exp(-1i*(0:sensor_number-1)*d*2*pi*sin(source_doa(2)*pi/180)/l)].';%阵列流型
s=10.^(snr/20)*exp(1i*2*pi*w*(0:N_x-1)/fs);%仿真信号
%rou = abs(mean(s(1,:).*conj(s(2,:)))/sqrt((mean(abs(s(1,:)).^2))*(mean(abs(s(2,:)).^2))));

%x=awgn(s,snr);
x=A*s+(1/sqrt(2))*(randn(sensor_number,N_x)+1i*randn(sensor_number,N_x));%加了高斯白噪声后的阵列接收信号

R=x*x'/snapshot_number;
%对数据协方差矩阵斜对角线上的元素进行平均,即Toeplitz化
dd=zeros(2*sensor_number-1);%15×15
for i=-(sensor_number-1):(sensor_number-1)
    c=sum(diag(R,i))/(sensor_number-abs(i));%每一对角线取平均
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
%对Toeplitz化后的协方差矩阵进行特征分解，得到信号子空间和噪声子空间
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

%利用MUSIC算法进行谱峰搜索，从而得到波到达方向
% searching_doa=-90:0.1:90;%线阵的搜索范围为-90~90度
searching_doa=-90:0.1:90;%线阵的搜索范围为-90~90度
 for kk=1:length(searching_doa)
   a_theta=exp(-1i*(0:sensor_number-1)'*2*pi*d*sin(pi*searching_doa(kk)/180)/l);
   Pmusic(kk)=1./abs((a_theta)'*Gn*a_theta);
 end
plot(searching_doa,10*log10(Pmusic));
%axis([-90 90 -90 90]);
xlabel('入射角/度');
ylabel('谱峰/dB');
% legend('TOEPLITZ MUSIC SPECTRUM');
title('Toeplitz近似MUSIC估计');
grid on;
