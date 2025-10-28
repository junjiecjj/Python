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
for i=I+1:I+L             %寻找第一副瓣的起始点
    if sout_dB(i)>sout_dB(i-1)
        j=i;
        break;
    else
        continue;
    end
end
for k=j:j+L              %寻找第一副瓣的最大值
    if sout_dB(k)<sout_dB(k-1)
        M=k-1;
        break;
    else
        continue;
    end
end
PSLR=sout_dB(M)