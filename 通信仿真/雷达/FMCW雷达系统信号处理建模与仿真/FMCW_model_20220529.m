%% 微信公众号：调皮的连续波
%% 知乎：调皮连续波
%% 作者：调皮哥
%% 时间：2022年5月
%% 功能：FMCW雷达发射信号、回波信号、混频、距离维FFT、速度维FFT建模仿真。
%%=========================================================================
clear all;
close all;
clc;
%% 雷达系统参数设置
fc= 77e9;             % 雷达工作频率 载频
c = 3e8;              % 光速

%% FMCW波形参数设置
maxR = 200;           % 雷达最大探测目标的距离
rangeRes = 1;         % 雷达的距离分率 
maxV = 20;            % 雷达最大检测目标的速度
B = c/(2*rangeRes);       % 发射信号带宽 (y-axis)  B = 150MHz
Nd=128;                          %chirp数量 
Nr=256;                        %ADC采样点数
Tchirp = 5 * 2 * maxR/c;  % 扫频时间 (x-axis), 5.5= sweep time should be at least 5 o 6 times the round trip time
endle_time=2.5e-6;          %空闲时间

vres=(c/fc)/(2*Nd*(Tchirp+endle_time));%速度分辨率

%% 用户自定义目标参数
r0 = rangeRes*90;  % 目标距离 rangeRes*距离通道号 距离通道号范围[1 256]
v0 = vres*5        % 目标速度 vres*速度通道号[1 128]

%%
slope = B / Tchirp;         %调频斜率
f_IFmax= (slope*2*maxR)/c ; %最高中频频率
f_IF=(slope*2*r0)/c ;       %当前中频频率
% Nr=1024*256;                %和频信号点数设置
Fs=Nr/Tchirp;                 %模拟信号采样频率

%%
t=linspace(0,Nd*Tchirp,Nr*Nd); %发射信号和接收信号的采样时间，在MATLAB中的模拟信号是通过数字信号无限采样生成的。
Tx=zeros(1,length(t)); %发射信号
Rx=zeros(1,length(t)); %接收信号
Mix = zeros(1,length(t)); %差频、差拍、拍频、中频信号
r_t=zeros(1,length(t));
td=zeros(1,length(t));
%% 动目标信号生成
for i=1:length(t)
    r_t(i) = r0 + v0*t(i); % 距离更新
    td(i) = 2*r_t(i)/c;    % 延迟时间
    
    Tx(i) = cos(2*pi*(fc*t(i) + (slope*t(i)^2)/2)); % 发射信号 实数信号
    Rx(i) = cos(2*pi*(fc*(t(i)-td(i)) + (slope*(t(i)-td(i))^2)/2)); %接收信号 实数信号
    
    if i<=1024
         freq(i)=fc+slope*i*Tchirp/Nr; %发射信号时频图 只取第一个chirp
         freq_echo(i)=fc+slope*i*Tchirp/Nr;%回波信号频谱延迟
    end

    Mix(i) = Tx(i).*Rx(i);%差频、差拍、拍频、中频信号
end

% figure;
% plot((abs(fft(MixIQ(1:1024)))));%查看宽带的和频信号 将chirp的点数改为1024*256即可看到有一个门信号，但注意计算机内存。
% xlabel('频率');
% ylabel('幅度');
% title('中频信号频谱');

%%
% %发射信号时域图
% figure;
% plot(Tx(1:1024));
% xlabel('点数');
% ylabel('幅度');
% title('TX发射信号时域图');
% 
% % %发射信号时频图
% figure;
% plot(t(1:1024),freq);
% xlabel('时间');
% ylabel('频率');
% title('TX发射信号时频图');
% 
% %接收信号时域图
% figure;
% plot(Rx(1:1024));
% xlabel('点数');
% ylabel('幅度');
% title('RX接收信号时域图');
% 
%接收信号与发射信号的时频图
% figure;
% plot(t(1:1024),freq);
% hold on;
% plot(td(1:1024)+t(1:1024),freq);
% xlabel('时间');
% ylabel('频率');
% title('接收信号与发射信号时频图');
% legend ('TX','RX');

%中频信号频谱 和频信号观察
%figure;
% plot(db(abs(fft(Mix(1:1024*256)))));%查看宽带的和频信号 将chirp的点数改为1024*256即可看到有一个门信号，但注意计算机内存。
% xlabel('频率');
% ylabel('幅度');
% title('中频信号频谱');

% figure;
% plot(db(abs(fft(Mix(1:1024)))));%查看宽带的和频信号 将chirp的点数改为1024*256即可看到有一个门信号，但注意计算机内存。
% xlabel('频率');
% ylabel('幅度');
% title('中频信号频谱');

%% 低通滤波 截止频率30MHz  采样频率120MHz
% Mix=lowpass(Mix(1:1024*256),30e6,120e6);
% plot(db(abs(fft(Mix(1:1024*256)))));
% xlabel('频率');
% ylabel('幅度');
% title('中频信号低通滤波器');

%reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
%Range and Doppler FFT respectively.
signal = reshape(Mix,Nr,Nd);

MixIQ=zeros(Nr,Nd);
for i=1:Nd

MixIQ(:,i) =hilbert(Mix(((i-1)*Nr+1):Nr*i)); % 希尔伯特变换 变为IQ正交信号

end

% figure;
% mesh(signal);
% xlabel('脉冲数')
% ylabel('距离门数');
% title('中频信号时域');

%% 距离维FFT
sig_fft1 = zeros(Nr,Nd);
for k=1:Nd
    sig_fft1(:,k)=fft(MixIQ(:,k));
end

sig_fft = abs(sig_fft1);
figure;
plot(sig_fft(:,1));
xlabel('距离（频率）');
ylabel('幅度')
title('第一个chirp的FTF结果')


%% 距离FFT结果谱矩阵
figure;
mesh(sig_fft);
ylabel('距离（频率）');
xlabel('chirp脉冲数')
zlabel('幅度')
title('距离维FTF结果')

%% 速度维FFT
sig_fft2 = zeros(Nr,Nd);
for k=1:Nr
    sig_fft2(k,:)=fft(sig_fft1(k,:));
end

% sig_fft2 = fft2(MixIQ);
RDM = abs(sig_fft2);
doppler_axis = linspace(0,128,Nd)*vres;
range_axis = linspace(0,256,Nr)*rangeRes;

figure;
mesh(doppler_axis,range_axis,RDM);
xlabel('多普勒通道'); ylabel('距离通道'); zlabel('幅度（dB）');
title('速度维FFT 距离多普勒谱');

%% END