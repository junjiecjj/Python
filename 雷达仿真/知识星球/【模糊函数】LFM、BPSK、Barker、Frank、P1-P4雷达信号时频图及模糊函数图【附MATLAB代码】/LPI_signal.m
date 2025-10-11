clc;clear all;close all;

%% 线性调频信号 LFM
fs = 500e3;
sLFM = phased.LinearFMWaveform('SampleRate',fs,...
    'SweepBandwidth',200e3,...
    'PulseWidth',1e-3,'PRF',1e3);

%时域图
figure
lfmwav = step(sLFM);
% lfmwav=awgn(lfmwav,-0);  %加噪声可要可不要
nsamp = size(lfmwav,1);
t = [0:(nsamp-1)]/fs;
plot(t*1000,real(lfmwav))
xlabel('时间（ms）')       
ylabel('幅度')
title('LFM信号时域图')
% saveas(gcf,'./img/LFM信号时域图.png')

%频域图
figure
nfft=nsamp;
Z = fft(lfmwav);
fr = [0:(nfft/2-1)]/nfft*fs;
plot(fr/1000,abs(Z(1:nfft/2)),'.-')
xlabel('频率（kHz）')
ylabel('幅度')
title('LFM信号频域图')
% saveas(gcf,'./img/LFM信号频域图.png')

%时频域图
figure
nsc = floor(nsamp/10);
nov = floor(nsc*0.9);
nff = max(256,2^nextpow2(nsc));
spectrogram(lfmwav,hamming(nsc),nov,nff,fs,'centered','yaxis');
title('LFM信号时频域图')
% saveas(gcf,'./img/LFM信号时频域图.png')

z=xcorr(lfmwav);
figure;
plot(1:999,z);
xlabel('采样点')
ylabel('幅度')
title('LFM信号自相关域图')
%模糊函数
[afmag,delay,doppler] = ambgfun(lfmwav,sLFM.SampleRate,sLFM.PRF);
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none'); 
axis tight; grid off;  colorbar;
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('LFM信号模糊函数')
shading interp;
% saveas(gcf,'./img/LFM信号模糊函数.png')

%% Frank码 相位编码信号
Rmax = 200;
Rres = 5;
c = 3e8;
prf = c/(2*Rmax);
bw = c/(2*Rres);
fs = 2*bw;

Frankwaveform = phased.PhaseCodedWaveform('Code','Frank','NumChips',25  ,...
    'SampleRate', fs,'ChipWidth',1/bw,'PRF',prf);
                                
%时域图
figure
Frankwav = Frankwaveform();
% Frankwav=awgn(Frankwav,30);  %加噪声可要可不要
nsamp = size(Frankwav,1);
t = [0:(nsamp-1)]/fs;  
plot(t*1000,real(Frankwav))
xlabel('时间（ms）')       
ylabel('幅度')
title('Frank相位编码信号时域图')
% saveas(gcf,'./img/Frank相位编码信号时域图.png')

%频域图
figure
nfft=12*nsamp;
Z = fft(Frankwav,nfft);
fr = [0:(nfft/2-1)]/nfft*fs;
plot(fr/1000,abs(Z(1:nfft/2)),'.-')
xlabel('频率（kHz）')
ylabel('幅度')
title('Frank相位编码信号频域图')
% saveas(gcf,'./img/Frank相位编码信号频域图.png')

z1=xcorr(Frankwav);
figure;
plot(1:(2*length(Frankwav)-1),z1);
xlabel('采样点')
ylabel('幅度')
title('Frank信号自相关域图')
%时频域图
figure
nsc = floor(nsamp/7);
nov = floor(nsc*0.9);
nff = max(256,2^nextpow2(nsc));
spectrogram(Frankwav,hamming(nsc),nov,nff,fs,'centered','yaxis');
title('Frank相位编码信号时频域图')
% saveas(gcf,'./img/Frank相位编码信号时频域图.png')

%模糊函数
[afmag,delay,doppler] = ambgfun(Frankwav,Frankwaveform.SampleRate,Frankwaveform.PRF);
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none'); 
axis tight; grid off;  colorbar;
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('Frank相位编码信号模糊函数')
shading interp;
% saveas(gcf,'./img/Frank相位编码信号模糊函数.png')

%% Barker码 相位编码信号
Rmax = 200;
Rres = 5;
c = 3e8;
prf = c/(2*Rmax);
bw = c/(2*Rres);
fs = 2*bw;

Barkerwaveform = phased.PhaseCodedWaveform('Code','Barker','NumChips',13  ,...
    'SampleRate', fs,'ChipWidth',1/bw,'PRF',prf);
                                
%时域图
figure
Barkerwav = Barkerwaveform();
% Barkerwav=awgn(Barkerwav,30);  %加噪声可要可不要
nsamp = size(Barkerwav,1);
t = [0:(nsamp-1)]/fs;  
plot(t*1000,real(Barkerwav))
xlabel('时间（ms）')       
ylabel('幅度')
title('Barker相位编码信号时域图')
% saveas(gcf,'./img/Barker相位编码信号时域图.png')

%频域图
figure
nfft=12*nsamp;
Z = fft(Barkerwav,nfft);
fr = [0:(nfft/2-1)]/nfft*fs;
plot(fr/1000,abs(Z(1:nfft/2)),'.-')
xlabel('频率（kHz）')
ylabel('幅度')
title('Barker相位编码信号频域图')
% saveas(gcf,'./img/Barker相位编码信号频域图.png')

%时频域图
figure
nsc = floor(nsamp/7);
nov = floor(nsc*0.9);
nff = max(256,2^nextpow2(nsc));
spectrogram(Barkerwav,hamming(nsc),nov,nff,fs,'centered','yaxis');
title('Barker相位编码信号时频域图')
% saveas(gcf,'./img/Barker相位编码信号时频域图.png')

%模糊函数
[afmag,delay,doppler] = ambgfun(Barkerwav,Barkerwaveform.SampleRate,Barkerwaveform.PRF);
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none'); 
axis tight; grid off;  colorbar;
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('Barker相位编码信号模糊函数')
shading interp;
% saveas(gcf,'./img/Barker相位编码信号模糊函数.png')



%% P1码 相位编码信号
Rmax = 200;
Rres = 5;
c = 3e8;
prf = c/(2*Rmax);
bw = c/(2*Rres);
fs = 2*bw;

P1waveform = phased.PhaseCodedWaveform('Code','P1','NumChips',25  ,...
    'SampleRate', fs,'ChipWidth',1/bw,'PRF',prf);
                                
%时域图
figure
P1wav = P1waveform();
% P1wav=awgn(P1wav,30);  %加噪声可要可不要
nsamp = size(P1wav,1);
t = [0:(nsamp-1)]/fs;  
plot(t*1000,real(P1wav))
xlabel('时间（ms）')       
ylabel('幅度')
title('P1相位编码信号时域图')
% saveas(gcf,'./img/P1相位编码信号时域图.png')

%频域图
figure
nfft=12*nsamp;
Z = fft(P1wav,nfft);
fr = [0:(nfft/2-1)]/nfft*fs;
plot(fr/1000,abs(Z(1:nfft/2)),'.-')
xlabel('频率（kHz）')
ylabel('幅度')
title('P1相位编码信号频域图')
% saveas(gcf,'./img/P1相位编码信号频域图.png')

%时频域图
figure
nsc = floor(nsamp/7);
nov = floor(nsc*0.9);
nff = max(256,2^nextpow2(nsc));
spectrogram(P1wav,hamming(nsc),nov,nff,fs,'centered','yaxis');
title('P1相位编码信号时频域图')
% saveas(gcf,'./img/P1相位编码信号时频域图.png')

%模糊函数
[afmag,delay,doppler] = ambgfun(P1wav,P1waveform.SampleRate,P1waveform.PRF);
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none'); 
axis tight; grid off;  colorbar;
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('P1相位编码信号模糊函数')
shading interp;
% saveas(gcf,'./img/P1相位编码信号模糊函数.png')


%% P2码 相位编码信号
Rmax = 200;
Rres = 5;
c = 3e8;
prf = c/(2*Rmax);
bw = c/(2*Rres);
fs = 2*bw;

P2waveform = phased.PhaseCodedWaveform('Code','P2','NumChips',16  ,...
    'SampleRate', fs,'ChipWidth',1/bw,'PRF',prf);
                                
%时域图
figure
P2wav = P2waveform();
% P2wav=awgn(P2wav,30);  %加噪声可要可不要
nsamp = size(P2wav,1);
t = [0:(nsamp-1)]/fs;  
plot(t*1000,real(P2wav))
xlabel('时间（ms）')       
ylabel('幅度')
title('P2相位编码信号时域图')
% saveas(gcf,'./img/P2相位编码信号时域图.png')

%频域图
figure
nfft=12*nsamp;
Z = fft(P2wav,nfft);
fr = [0:(nfft/2-1)]/nfft*fs;
plot(fr/1000,abs(Z(1:nfft/2)),'.-')
xlabel('频率（kHz）')
ylabel('幅度')
title('P2相位编码信号频域图')
% saveas(gcf,'./img/P2相位编码信号频域图.png')

%时频域图
figure
nsc = floor(nsamp/7);
nov = floor(nsc*0.9);
nff = max(256,2^nextpow2(nsc));
spectrogram(P2wav,hamming(nsc),nov,nff,fs,'centered','yaxis');
title('P2相位编码信号时频域图')
% saveas(gcf,'./img/P2相位编码信号时频域图.png')

%模糊函数
[afmag,delay,doppler] = ambgfun(P2wav,P2waveform.SampleRate,P2waveform.PRF);
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none'); 
axis tight; grid off;  colorbar;
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('P2相位编码信号模糊函数')
shading interp;
% saveas(gcf,'./img/P2相位编码信号模糊函数.png')

%% P3码 相位编码信号
Rmax = 200;
Rres = 5;
c = 3e8;
prf = c/(2*Rmax);
bw = c/(2*Rres);
fs = 2*bw;

P3waveform = phased.PhaseCodedWaveform('Code','P3','NumChips',25  ,...
    'SampleRate', fs,'ChipWidth',1/bw,'PRF',prf);
                                
%时域图
figure
P3wav = P3waveform();
% P3wav=awgn(P3wav,30);  %加噪声可要可不要
nsamp = size(P3wav,1);
t = [0:(nsamp-1)]/fs;  
plot(t*1000,real(P3wav))
xlabel('时间（ms）')       
ylabel('幅度')
title('P3相位编码信号时域图')
% saveas(gcf,'./img/P3相位编码信号时域图.png')

%频域图
figure
nfft=12*nsamp;
Z = fft(P3wav,nfft);
fr = [0:(nfft/2-1)]/nfft*fs;
plot(fr/1000,abs(Z(1:nfft/2)),'.-')
xlabel('频率（kHz）')
ylabel('幅度')
title('P3相位编码信号频域图')
% saveas(gcf,'./img/P3相位编码信号频域图.png')

%时频域图
figure
nsc = floor(nsamp/7);
nov = floor(nsc*0.9);
nff = max(256,2^nextpow2(nsc));
spectrogram(P3wav,hamming(nsc),nov,nff,fs,'centered','yaxis');
title('P3相位编码信号时频域图')
% saveas(gcf,'./img/P3相位编码信号时频域图.png')

%模糊函数
[afmag,delay,doppler] = ambgfun(P3wav,P3waveform.SampleRate,P3waveform.PRF);
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none'); 
axis tight; grid off;  colorbar;
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('P3相位编码信号模糊函数')
shading interp;
% saveas(gcf,'./img/P3相位编码信号模糊函数.png')



%% P4码 相位编码信号
Rmax = 200;
Rres = 5;
c = 3e8;
prf = c/(2*Rmax);
bw = c/(2*Rres);
fs = 2*bw;

P4waveform = phased.PhaseCodedWaveform('Code','P4','NumChips',25  ,...
    'SampleRate', fs,'ChipWidth',1/bw,'PRF',prf);
                                
%时域图
figure
P4wav = P4waveform();
% P4wav=awgn(P4wav,30);  %加噪声可要可不要
nsamp = size(P4wav,1);
t = [0:(nsamp-1)]/fs;  
plot(t*1000,real(P4wav))
xlabel('时间（ms）')       
ylabel('幅度')
title('P4相位编码信号时域图')
% saveas(gcf,'./img/P4相位编码信号时域图.png')

%频域图
figure
nfft=12*nsamp;
Z = fft(P4wav,nfft);
fr = [0:(nfft/2-1)]/nfft*fs;
plot(fr/1000,abs(Z(1:nfft/2)),'.-')
xlabel('频率（kHz）')
ylabel('幅度')
title('P4相位编码信号频域图')
% saveas(gcf,'./img/P4相位编码信号频域图.png')

%时频域图
figure
nsc = floor(nsamp/7);
nov = floor(nsc*0.9);
nff = max(256,2^nextpow2(nsc));
spectrogram(P4wav,hamming(nsc),nov,nff,fs,'centered','yaxis');
title('P4相位编码信号时频域图')
% saveas(gcf,'./img/P4相位编码信号时频域图.png')

%模糊函数
[afmag,delay,doppler] = ambgfun(P4wav,P4waveform.SampleRate,P4waveform.PRF);
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none'); 
axis tight; grid off;  colorbar;
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('P4相位编码信号模糊函数')
shading interp;
% saveas(gcf,'./img/P4相位编码信号模糊函数.png')

%% Costas 编码
% Costas_code=[1, 2, 4, 8, 16, 32,...
%              27, 17, 34, 31, 25, ...
%              13, 26, 15, 30, 23, ...
%              9, 18, 36, 35, 33,...
%              29, 21, 5, 10, 20, ...
%              3, 6, 12, 24, 11, ...
%              22, 7, 14, 28, 19];
Costas_code=[1:1:10];
Costas_code=Costas_code(randperm(length(Costas_code)));

base_freq=1e4;
N=length(Costas_code);
fs=N*base_freq*2;

Costas_signal=[];
for i =1:N
    frequencyOffset=base_freq*Costas_code(i);
     pulse_LFM_signal = phased.LinearFMWaveform('SampleRate',fs,...
    'SweepBandwidth',5e3,'FrequencyOffset',frequencyOffset,...
    'PulseWidth',1e-3,'PRF',1e3);
    Costas_signal(i,:)=pulse_LFM_signal();
end
size_signal=size(Costas_signal);
Costas_signal=reshape(Costas_signal.',[1,N*size_signal(2)]);

%时域图
figure
nsamp = length(Costas_signal)/36;
t = [0:(nsamp-1)]/fs;  
plot(t*1000,real(Costas_signal(1:nsamp)))
xlabel('时间（ms）')       
ylabel('幅度')
title('单个码元内Costas跳频编码信号时域图')
% saveas(gcf,'./img/Costas跳频编码信号时域图.png')

%频域图
figure
nsamp = length(Costas_signal);
nfft=12*nsamp;
Z = fft(Costas_signal,nfft);
fr = [0:(nfft/2-1)]/nfft*fs;
plot(fr/1000,abs(Z(1:nfft/2)),'.-')
xlabel('频率（kHz）')
ylabel('幅度')
title('Costas跳频编码信号频域图')
% saveas(gcf,'./img/Costas跳频编码信号频域图.png')

%时频域图
figure
spectrogram(Costas_signal,hamming(160),120,512,fs,'centered','yaxis');
title('Costas跳频编码信号时频域图')
% saveas(gcf,'./img/Costas跳频编码信号时频域图.png')

%模糊函数
[afmag,delay,doppler] = ambgfun(Costas_signal, pulse_LFM_signal.SampleRate, pulse_LFM_signal.PRF);
figure
surf(delay*1e6,doppler/1e3,afmag,'LineStyle','none'); 
axis tight; grid off;  colorbar;
xlabel('Delay \tau (us)');ylabel('Doppler f_d (kHz)');
title('Costas跳频编码信号模糊函数')
shading interp;
% saveas(gcf,'./img/Costas跳频编码信号模糊函数.png')

