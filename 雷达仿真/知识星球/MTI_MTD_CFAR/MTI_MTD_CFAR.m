close all; 
clear all; 
clc;
%% 信号参数
C=3.0e8;  
RF=5e9;  %RF
Lambda=C/RF;
PulseNumber=32;   %回波脉冲数 
BandWidth=2.0e6;  %发射信号带宽
TimeWidth=40.0e-6; %发射信号时宽
PRT=200e-6;   
PRF=1/PRT;
Fs=10.0e6;  %采样频率
AWGNpower = 0;%dB

SampleNumber=fix(Fs*PRT);%计算一个脉冲周期的采样点数；
TotalNumber=SampleNumber*PulseNumber;%总的采样点数；
BlindNumber=fix(Fs*TimeWidth);%计算一个脉冲周期的盲区-遮挡样点数；

%% 目标参数
TargetNumber=3;%目标个数
SigPower(1:TargetNumber)=[1 1 1];                                       %目标功率,无量纲
TargetDistance(1:TargetNumber)=[5000 15000 20000];                      %目标距离,单位m
DelayNumber(1:TargetNumber)=fix(Fs*2*TargetDistance(1:TargetNumber)/C); %把目标距离换算成采样点（距离门）
TargetVelocity(1:TargetNumber)=[0 50 200];                              %目标径向速度 单位m/s
TargetFd(1:TargetNumber)=2*TargetVelocity(1:TargetNumber)/Lambda;       %计算目标多卜勒
%% 信号产生
 number=fix(Fs*TimeWidth);%回波的采样点数=脉压系数长度=暂态点数目+1
if rem(number,2)~=0
   number=number+1;
end   
for i=-fix(number/2):fix(number/2)-1
  Chirp(i+fix(number/2)+1)=exp(j*(pi*(BandWidth/TimeWidth)*(i/Fs)^2));
  
end
coeff=conj(fliplr(Chirp));
%回波串
SignalAll=zeros(1,TotalNumber);%所有脉冲的信号,先填0
for k=1:TargetNumber% 依次产生各个目标
   SignalTemp=zeros(1,SampleNumber);% 一个脉冲
   SignalTemp(DelayNumber(k)+1:DelayNumber(k)+number)=sqrt(SigPower(k))*Chirp;%一个脉冲的1个目标（未加多普勒速度）
   Signal=zeros(1,TotalNumber);
   for i=1:PulseNumber
      Signal((i-1)*SampleNumber+1:i*SampleNumber)=SignalTemp;
   end
   FreqMove=exp(1j*2*pi*TargetFd(k)*(0:TotalNumber-1)/Fs);%目标的多普勒速度*时间=目标的多普勒相移
   Signal=Signal.*FreqMove;
   SignalAll=SignalAll+Signal;
end

Echo = awgn(SignalAll,AWGNpower);

for i=1:PulseNumber   %在接收机闭锁期,接收的回波为0
      Echo((i-1)*SampleNumber+1:(i-1)*SampleNumber+number)=0;
end
pc_time0=conv(Echo,coeff);

% 频域脉压
nfft = 2^nextpow2(length(Echo));%FFT点数
Echo_fft=fft(Echo,nfft);
coeff_fft=fft(coeff,nfft);
pc_fft=Echo_fft.*coeff_fft;
pc_freq0=ifft(pc_fft);
pc_freq1=pc_freq0(number:TotalNumber+number-1);%去掉暂态点 number-1个,后填充点若干(2048-number+1-TotalNumber=45个)

%按照脉冲号、距离门号重排数据
for i=1:PulseNumber
      pc(i,1:SampleNumber)=pc_freq1((i-1)*SampleNumber+1:i*SampleNumber);
end

%% MTI
mti = zeros(PulseNumber-1,size(pc,2));
for i=1:PulseNumber-1  %滑动对消，少了一个脉冲
   mti(i,:)=pc(i+1,:)-pc(i,:);
end
figure;
  xmti = C/2*(0:1/Fs:PRT-1/Fs);
  ymti = 1:PulseNumber-1;
mesh(xmti,ymti,abs(mti));title('MTI 结果');
  xlabel('距离/m');

figure;
subplot(211)
plot(xmti,abs(pc(1,:)));grid on;
title('脉冲压缩结果');xlabel('距离/m');
subplot(212)
plot(xmti,abs(mti(1,:)));grid on;
title('MTI一次对消结果');xlabel('距离/m');

%% MTD
mtd=zeros(PulseNumber,SampleNumber);
for i=1:SampleNumber
   buff(1:PulseNumber)=pc(1:PulseNumber,i);
   buff_fft=fft(buff);
   mtd(1:PulseNumber,i)=buff_fft(1:PulseNumber)';
end
  
  xmtd = C/2*(0:1/Fs:PRT-1/Fs);
  ymtd = Lambda/2*(0:PRF/PulseNumber:PRF-PRF/PulseNumber);
  figure;
  mesh(xmtd,ymtd,abs(mtd));title('MTD  结果');
  xlabel('距离/m');ylabel('速度/ms^-^1')
  xlim([min(xmtd) max(xmtd)]);
  ylim([min(ymtd) max(ymtd)]);
  
mtd_result = max(abs(mtd));


%% CFAR
Pfa = 1e-6;
alpha = SampleNumber*(Pfa^(-1/SampleNumber)-1);
% alpha = 1;
num = 60;
protect = 20;

cfar_result(1,1)=mean(mtd_result(1,2:(num+1)));%第1点恒虚警处理时噪声均值由其后面的num点的噪声决定
for i=2:num  %%%%第2点到第num点的恒虚警的噪声均由其前面和后面的num点的噪声共同决定
    cfar_result(1,i)=(mean(mtd_result(1,1:i-1))+mean(mtd_result(1,i+1:i+num)))/2;
end
for i=(num+1):SampleNumber-(num+1)
%正常的数据点恒虚警处理的噪声均值由其前面和后面各num点噪声共同决定
    cfar_result(1,i)=max(mean(mtd_result(1,i-num:i-protect)),mean(mtd_result(1,i+protect:i+num)));
end
for i=SampleNumber-num:SampleNumber-1
%倒数第num点到倒数第2点恒虚警处理的噪声均值由其前面num点和后面的噪声共同决定
    cfar_result(1,i)=(mean(mtd_result(1,i-num:i-1))+mean(mtd_result(1,(i+1):SampleNumber)))/2;
end
%%%最后一点的恒虚警处理的噪声均值由其前面的num点的噪声决定 
cfar_result(1,SampleNumber)=mean(mtd_result(1,SampleNumber-num:SampleNumber-1));
s_result=zeros(1,SampleNumber);
for i=1:SampleNumber
    temp = mtd_result(1,i);
    temp1 = alpha*cfar_result(1,i);
    if mtd_result(1,i)>=alpha*cfar_result(1,i)
    s_result(1,i)=mtd_result(1,i);
    else s_result(1,i)=0;
    end
end


figure;
subplot(211)
plot(xmtd,mtd_result);  hold on;
plot(xmtd,alpha*cfar_result,'r');hold off;
xlim([min(xmtd) max(xmtd)]);
xlabel('距离/m');
title('MTD处理后求模结果(信号最大通道)');
grid on;

subplot(212);
plot(xmtd,s_result);
xlim([min(xmtd) max(xmtd)]);
xlabel('距离/m');
title('恒虚警处理结果');
grid on;
 