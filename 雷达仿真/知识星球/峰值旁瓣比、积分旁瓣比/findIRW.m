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

maxdata=max(sout_dB);
x1=find(sout_dB>=maxdata-3);
I=length(x1);
IRW=I*(1/fs)