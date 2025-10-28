close all;clear all;
%% LFM信号的参数
T=10e-6;                          %信号时宽
B=30e6;                           %信号带宽
K=B/T;                            %线性调频系数
fc=0;                             %信号载频
a=20;                              %过采样因子
fs=a*B;Ts=1/fs;                   %采样率Fs
t0=0;                             %时延
tc=0;                        %tc=0为基带信号，tc不为0是为非基带信号
N=T/Ts;                           %采样点数
%% 信号生成
t=linspace(-T/2,T/2,N);
st=exp(1i*pi*K*(t-tc).^2);         %调频信号
ht=exp(-1i*pi*K*(t+tc).^2);        %匹配滤波器
%% 匹配输出
sout=conv(st,ht,'same');
sout_dB=20*log10(abs(sout)/max(abs(sout)));%输出归一化的脉压后的幅度（dB）

L=length(sout_dB);
[maxdata,I]=max(sout_dB);
for i=I:I+L             %寻找第一零点
    if sout_dB(i+1)>sout_dB(i)
        j=i;
        break;
    else
        continue;
    end
end
p1=sum((10.^(sout_dB(2*I-j:j)/20)).^2*(1/fs));%主瓣功率积分
p=sum((10.^(sout_dB(round(-11/B*fs)+I:round(11/B*fs)+I)/20)).^2*(1/fs));%信号总功率
ISLR=10*log10((p-p1)/p1)