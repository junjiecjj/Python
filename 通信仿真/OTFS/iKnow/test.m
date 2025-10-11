
clc;
clear all;
close all;
%% 参数设置
fs=100; %采样率
tp=0.2; %脉冲宽度
Tr=1;%脉冲重复周期
M=10;
t1=-Tr/2:1/fs:Tr/2-1/fs;
t=-M*Tr/2:1/fs:M*Tr/2-1/fs;
N=length(t);
u=zeros(1,length(t));
s1=zeros(1,length(t));
%% 相参脉冲串
for i=0:M-1;
    s1=rectpuls(t-i*Tr-tp/2,tp); 
    u=u+s1;
end
figure(1)
plot(t,u)
%% 模糊函数图
fa_i=linspace(-1/tp,1/tp,N);      %多普勒频移序列
tao_i=linspace(-M*Tr,M*Tr,N);        %时域延时序列
[Tao_i,Fa_i]=meshgrid(tao_i,fa_i);
f=(-N/2:N/2-1)*fs/N;
U=fftshift(fft(u,N));   %信号FFT
U1=zeros(N,N);
u1=zeros(N,N);
U_fin=zeros(N,N);
for i=1:N
    u1(i,:)=u.*exp(1j*2*pi*fa_i(i).*t);
    U1(i,:)=fftshift(fft(u1(i,:),N));
    U_fin(i,:)=U1(i,:).*conj(U);
    u_fin(i,:)=fftshift(ifft(U_fin(i,:),N));
end
figure(2)
mesh(Tao_i,Fa_i,abs(u_fin))
xlabel('时延/s');
ylabel('多普勒频移/Hz');
zlabel('相参脉冲串模糊函数三维图');
figure(3)
contour(Tao_i,Fa_i,abs(u_fin))
xlabel('时延/s');
ylabel('多普勒频移/Hz');
zlabel('相参脉冲串模糊函数等值线图');
%% 0-fd切割面
U_2=U.*conj(U);
u_2=fftshift(ifft(U_2,N));
figure(4)
plot(t,abs(u_2)/max(abs(u_2)))
grid
xlabel('时延/s');
ylabel('多普勒频移/Hz');
zlabel('相参脉冲串零频移切面图');
%% 0-tao切割面
figure(5)
plot(fa_i,abs(u_fin(:,N/2+1)))
grid
xlabel('时延/s');
ylabel('多普勒频移/Hz');
zlabel('相参脉冲串零延时切面图');