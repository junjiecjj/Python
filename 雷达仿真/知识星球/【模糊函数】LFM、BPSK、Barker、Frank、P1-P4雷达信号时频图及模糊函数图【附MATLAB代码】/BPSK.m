%*****************BPSK***************************
clear all;
close all;
clc;
% s=randi([0 1],1,1000);
s=[1 1 0 1 0 0 1 0];
f=50e3;                     % 采样频率;%载波信号频率
PRF=1e3;
N=length(s);
%% 调制
t=0:2*pi/99:2*pi;
cp1=2*s-1;%双极性非归零码
bit=[];
cp=[];
mod=[];
for n=1:N
    bit=[bit s(n).*ones(1,100)];
    cp=[cp cp1(n).*ones(1,100)];
    c=cos(f*t);
    mod=[mod c];
end
bpsk_mod=cp.*mod;
%% 时域图
subplot(2,1,1);
plot(bit,'LineWidth',1.5);grid on;
title('Binary Signal');
axis([0 100*length(s) -2 2]);

subplot(2,1,2);
plot(bpsk_mod,'LineWidth',1.5);grid on;
title('BPSK modulation');
axis([0 100*length(s) -2 2]);
xlabel('1            1            0            1            0            0            1            0')

%% 频域图
figure
nfft=100*N;                 %做FFT的点数就是信号的点数
Z = fftshift(fft(bpsk_mod));            %对信号做FFT
fr = [0:(nfft/2-1)]/nfft*f;        %fft的频率范围
plot(fr*1e-3,abs(Z(nfft/2+1:end)),'.-')             %画fft出来的频谱图
xlabel('频率（kHz）')
ylabel('幅度')
title('BPSK信号频域图')

z1=xcorr(bpsk_mod);
figure;
plot(1:(2*length(bpsk_mod)-1),z1);
xlabel('采样点')
ylabel('幅度')
title('BPSK信号自相关域图')
%% 时频域图
figure
nsc = floor(100*N/10);          %做STFT的窗长
nov = floor(nsc*0.9);           %两个窗长的覆盖部分的长度
nff = max(256,2^nextpow2(nsc));     %做STFT的点数
spectrogram(bpsk_mod,hamming(nsc),nov,nff,f,'centered','yaxis');     %画出频谱图
title('BPSK信号时频域图')
xlabel('1            1            0            1            0            0            1            0') 

%% 模糊函数
[afmag,delay,doppler] = ambgfun(bpsk_mod,f,PRF);       %对信号计算模糊函数 直接调用现成函数就行了 给一些参数就行
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none');       %画三维的模糊函数图
axis tight; grid off;  colorbar;            %设置坐标轴格式 把格点去掉 显示colorbar
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('BPSK信号模糊函数')
shading interp;