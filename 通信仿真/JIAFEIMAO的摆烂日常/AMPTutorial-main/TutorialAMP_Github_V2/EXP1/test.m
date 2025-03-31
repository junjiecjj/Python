clc;clear all;close all;



clc
clear all
close all
%**************参数配置*********************
Tp=200e-6;          %发射脉冲宽度s
B=1e6;           %调频带宽Hz
Ts=0.5e-6;       %采样时钟s
R0=[80e3,85e3];      %目标的距离m
vr=[0,0];            %目标速度
SNR=[20 10];         %信噪比
Rmin=20e3;           %采样的最小距离
Rrec=150e3;          %接收距离窗的大小
bos=2*pi/0.03;       %波数2*pi/λ。

%*********************************************
mu=B/Tp;            %调频率
c=3e8;              %光速m/s
M=round(Tp/Ts);
t1=(-M/2+0.5:M/2-0.5)*Ts;       %时间矢量
NR0=ceil(log2(2*Rrec/c/Ts));NR1=2^NR0;    %计算FFT的点数
lfm=exp(j*pi*mu*t1.^2);                   %信号复包络
lfm_w=lfm.*hanning(400)';
gama=(1+2*vr./c).^2;                      
sp=0.707*(randn(1,NR1)+j*randn(1,NR1));        %噪声
for k= 1:length(R0)
    NR=fix(2*(R0(k)-Rmin)/c/Ts);
    spt=(10^(SNR(k)/20))*exp(-j*bos*R0(k))*exp(j*pi*mu*gama(k)*t1.^2);     %信号
    sp(NR:NR+M-1)=sp(NR:NR+M-1)+spt;
end
spf=fft(sp,NR1);
lfmf=fft(lfm,NR1);      %未加窗
lfmf_w=fft(lfm_w,NR1);      %加窗
y=abs(ifft(spf.*conj(lfmf),NR1)/NR0);
y1=abs(ifft(spf.*conj(lfmf_w),NR1)/NR0);   %加窗
figure;
plot(real(sp));grid on;
xlabel('时域采样点');
figure;
plot(t1*1e6,real(lfm));grid on;
xlabel('时间/us')
ylabel('匹配滤波系数实部')
figure
plot((0:NR1-1)/10,20*log10(y));grid on;
xlabel("距离/km")
ylabel("脉压输出/dB");
title("脉冲压缩结果（未加窗）")

figure
plot((0:NR1-1)/10,20*log10(y1));grid on;
xlabel("距离/km")
ylabel("脉压输出/dB");
title("脉冲压缩结果（加窗）")