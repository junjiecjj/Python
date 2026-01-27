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


%% 加窗效应
M=length(ht);%窗的长度
w=hanning(M);%加的窗函数的类型
sout_win=conv(st,(ht.*w'),'same');%加窗后的输出
sout_dB_win=20*log10(abs(sout_win)/max(abs(sout_win)));%加窗后输出归一化的脉压后的幅度（dB）

%%
figure;
subplot(221);plot(t*1e6,real(st));grid on;
title('信号实部');xlabel('时间/us');ylabel('信号实部幅度');
subplot(222);plot(t*1e6,imag(st));grid on;
title('信号虚部');xlabel('时间/us');ylabel('信号虚部幅度');
subplot(223);plot(t*1e6,real(ht));grid on;
title('滤波器实部');xlabel('时间/us');ylabel('滤波器实部幅度');
subplot(224);plot(t*1e6,imag(ht));grid on;
title('滤波器虚部');xlabel('时间/us');ylabel('滤波器虚部幅度');
figure;
plot(t*1e6,sout_dB_win);grid on;
title('基带信号脉压结果');xlabel('时间/us');ylabel('压缩后的幅度（dB）');
zoom xon