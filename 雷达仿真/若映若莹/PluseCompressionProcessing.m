% 仿真1：雷达信号脉宽10us，中心频率10MHz，调频带宽2MHz，回波中频采样率为40MHz，假设在13.5km处有一个点目标，仿真器脉冲压缩处理。
% https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247488859&idx=1&sn=a58eb3c9005760af3203497367ea2c5b&chksm=ce760bdf90b7b0195ab47040032673af4c814d92449380c0d1278a6f571d3ebb4041862f4120&mpshare=1&scene=1&srcid=0127zz2sRpdmULHaVYBYmRYj&sharer_shareinfo=93b1ad23ef065ae62b5e0cf5b7120660&sharer_shareinfo_first=93b1ad23ef065ae62b5e0cf5b7120660&exportkey=n_ChQIAhIQG9AkXVcl8LWvHAIwqAm5zBKYAgIE97dBBAEAAAAAADqTBq6t%2F0QAAAAOpnltbLcz9gKNyK89dVj0teDqhzrLuDWBdWBFW%2BEibmOhs66mRpeEww9vSlqg8LgIUkWMlQTBkaUqIi9nhEYhCErxx%2Fx0Iuv9WQdRbmqmFxglJtGyRIiAQiAsFJL7z3jtnOx20EN1RP%2F7lzT%2BtZNsEWJyhbr9RLO1N041HVF716nrWE1ZthMuzBTvJGhqBIQbmw1%2BiHb9cV64Mh44PuQkjzSEUHHJo%2B6RnxuQJ1pFUYOWzgLgBCkaInJZMI7qvx2aiYE4Og1Z5pC%2BIvSo3cxKYMcluYuHZWov%2Fv8K3l1WAcyd04gbUSL6aC2IwPvYcM6wJA1yLQ1s5WT%2FyXPfte1UT2s%3D&acctmode=0&pass_ticket=Mhclruh%2BNGii8iKBcKmCCz7hu5Tj9z%2F0MPyfIofZib9uRLUQEl5dzsnUkFJupMeH&wx_header=0#rd
clc
clear all
close all
t = 10e-6;     %脉冲宽度
fs = 40e6;     %采样率
ts = 1 / fs;   %采样间隔
fc = 9e6;      %载频
f0 =10e6;      
B=2e6;
ft=0:1/fs:t-1/fs;
N=length(ft);
k=B/fs*2*pi/max(ft);
y=modulate(ft,fc,fs,'fm',k);
y_fft_result=fft(y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%正交解调%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=0:N-1;
local_oscillator_i=cos(n*f0/fs*2*pi);
local_oscillator_q=sin(n*f0/fs*2*pi);
fbb_i=local_oscillator_i.*y;
fbb_q=local_oscillator_q.*y;
window=chebwin(51,40);
[b,a]=fir1(50,2*B/fs,window);
fbb_i=[fbb_i,zeros(1,25)];
fbb_q=[fbb_q,zeros(1,25)];
fbb_i=filter(b,a,fbb_i);
fbb_q=filter(b,a,fbb_q);
fbb_i=fbb_i(26:end);
fbb_q=fbb_q(26:end);
fbb=fbb_i+j*fbb_q;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%产生理想线性调频脉冲压缩系数%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=4096;
D=B*t;
match_filter=ts*fliplr(conj(fbb))*sqrt(D)*2/t;
match_filter_fft=fft(match_filter,M);
figure,
subplot(2,1,1),plot(real(match_filter_fft)),title('脉冲压缩系数（实部）');
subplot(2,1,2),plot(imag(match_filter_fft)),title('脉冲压缩系数（虚部）');
%%%%%%%%%%%%%%%%%%%%%%%%%%%产生理想回波信号%%%%%%%%%%%%%%%%%%%%%%%%%%
t1=100e-6;
signal=[zeros(1,floor((t1-2*t)/ts)),y,zeros(1,floor(t/ts))];
n=0:t1/ts-1;
local_oscillator_i=cos(n*f0/fs*2*pi);
local_oscillator_q=sin(n*f0/fs*2*pi);
fbb_i=local_oscillator_i.*signal;
fbb_q=local_oscillator_q.*signal;
window=chebwin(51,40);
[b,a]=fir1(50,2*B/fs,window);
fbb_i=[fbb_i,zeros(1,25)];
fbb_q=[fbb_q,zeros(1,25)]
fbb_i=filter(b,a,fbb_i);
fbb_q=filter(b,a,fbb_q);
fbb_i=fbb_i(26:end);
fbb_q=fbb_q(26:end);
signal=fbb_i+j*fbb_q;
%%%%%%%%%%%%%%%%%%%%%%%%%%%脉冲压缩处理%%%%%%%%%%%%%%%%%%%%%%%%%%
signal_fft=fft(signal,M);
pc_result_fft=signal_fft.*match_filter_fft;
pc_result=ifft(pc_result_fft,M);
figure,
subplot(2,1,1),plot((0:ts:t1-ts),signal),title('解调后');
subplot(2,1,2),plot((0:ts:length(signal)*ts-ts),abs(pc_result(1:length(signal)))),
xlabel('时间，单位：s'),title('回波脉冲压缩处理结果');


% 仿真2：雷达信号脉宽为0.5us，中心频率10MHz的13位巴克码2PSK信号，回波中频采样率为40MHz,假设13.5km处有一个点目标，仿真雷达脉冲压缩处理。

clc
clear all
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%产生雷达发射信号%%%%%%%%%%%%%%%%%%%%%%%%%%
code=[1,1,1,1,1,-1,-1,1,1,-1,1,-1,1]; %13位巴克码
tao=0.5e-6;%脉冲宽度10μs
f0=10e6;
fs=40e6;  %采样频率100MHz
ts=1/fs;
t_tao=0:1/fs:tao-1/fs;
n=length(code);
N=length(t_tao);
pha=0;
t=0:1/fs:13*tao-1/fs;
s=zeros(1,length(t));
for i=1:n;
    if code(i)==1
        pha=pi;
    else 
        pha=0;
    end
        s(1,(i-1)*N+1:i*N)=cos(2*pi*f0*t_tao+pha);
end
figure(1),subplot(2,1,1),plot(t,s),xlabel('t(单位:s)'),title('混合调制信号(13 位巴克码+线性调频)');
s_fft_result=abs(fft(s(1:N)));
subplot(2,1,2),plot((0:fs/N:fs/2-fs/N),abs(s_fft_result(1:N/2))),xlabel('频率(单位:Hz)'),title('码内信号频谱');
%________正交解调________%
N=length(t);
n=0:N-1;
local_oscillator_i=cos(n*f0/fs*2*pi);%i路本振信号
local_oscillator_q=sin(n*f0/fs*2*pi);%q路本振信号
fbb_i=local_oscillator_i.*s;%i路解调
fbb_q=local_oscillator_q.*s;%q路解调
window=chebwin(51,40);
[b,a]=fir1(50, 0.5,window);
fbb_i=[fbb_i,zeros(1,25)];
fbb_q=[fbb_q,zeros(1,25)];
fbb_i=filter(b,a,fbb_i);
fbb_q=filter(b,a,fbb_q);
fbb_i=fbb_i(26:end);%截取有效信息
fbb_q=fbb_q(26:end);%截取有效信息
fbb=fbb_i+j*fbb_q;
%-------产生理想线性调频脉冲压缩匹配系数-------%
M=131072;%因为回波信号数据长度为3600点,所以利用FFT,做4096点FFT
t1=100e-6;
t=tao*length(code);
match_filter=2*ts*fliplr(conj(fbb))*2/t;
match_filter_fft=fft(match_filter,M);%第一次脉冲压缩处理匹配系数
figure(2),
subplot(2,1,1),plot(real(match_filter_fft)),title('脉冲压缩系数(实部)');
subplot(2,1,2),plot(imag(match_filter_fft)),title('脉冲压缩系数(虚部)');
%%%%%%%%%%产生理想点目标回波信号%%%%%%%%%%
signal=[zeros(1,(t1-2*t)/ts),s,zeros(1,t/ts)];
%%%%%%% 正交解调%%%%%%%
N=length(signal);
n=0:N-1;
local_oscillator_i=cos(n*f0/fs*2*pi);%i路本振信号
local_oscillator_q=sin(n*f0/fs*2*pi);%q路本振信号
fbb_i=local_oscillator_i.*signal;%i路解调
fbb_q=local_oscillator_q.*signal;%q路解调
window=chebwin(51,40);%这是采用50阶cheby窗的FIR低通滤波器
[b,a]=fir1(50, 0.5,window);
fbb_i=[fbb_i,zeros(1,25)];
fbb_q=[fbb_q,zeros(1,25)];
fbb_i=filter(b,a,fbb_i);
fbb_q=filter(b,a,fbb_q);
fbb_i=fbb_i(26:end);%截取有效信息
fbb_q=fbb_q(26:end);%截取有效信息
signal=fbb_i+j*fbb_q;
clear fbb_i;
clear fbb_q;
%%%%%%%%%%%%%%%%%%脉压处理%%%%%%%%%%%%%%%%%%
signal_fft=fft(signal,M);
pc_result_fft=signal_fft.*match_filter_fft;
pc_result=ifft(pc_result_fft,M);
figure(3),
subplot(2,1,1),plot((0:ts:t1-ts),signal),xlabel('时间,单位:s'),
title('回波信号（解调后）');
subplot(2,1,2),plot((0:ts:length(signal)*ts-ts),abs(pc_result(1:length(signal)))),
xlabel('时间,单位:s'),
title('回波脉冲压缩处理结果');