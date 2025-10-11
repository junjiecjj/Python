close all; 
clear all; 
clc;
%% �źŲ���
C=3.0e8;  
RF=5e9;  %RF
Lambda=C/RF;
PulseNumber=32;   %�ز������� 
BandWidth=2.0e6;  %�����źŴ���
TimeWidth=40.0e-6; %�����ź�ʱ��
PRT=200e-6;   
PRF=1/PRT;
Fs=10.0e6;  %����Ƶ��
AWGNpower = 0;%dB

SampleNumber=fix(Fs*PRT);%����һ���������ڵĲ���������
TotalNumber=SampleNumber*PulseNumber;%�ܵĲ���������
BlindNumber=fix(Fs*TimeWidth);%����һ���������ڵ�ä��-�ڵ���������

%% Ŀ�����
TargetNumber=3;%Ŀ�����
SigPower(1:TargetNumber)=[1 1 1];                                       %Ŀ�깦��,������
TargetDistance(1:TargetNumber)=[5000 15000 20000];                      %Ŀ�����,��λm
DelayNumber(1:TargetNumber)=fix(Fs*2*TargetDistance(1:TargetNumber)/C); %��Ŀ����뻻��ɲ����㣨�����ţ�
TargetVelocity(1:TargetNumber)=[0 50 200];                              %Ŀ�꾶���ٶ� ��λm/s
TargetFd(1:TargetNumber)=2*TargetVelocity(1:TargetNumber)/Lambda;       %����Ŀ��ಷ��
%% �źŲ���
 number=fix(Fs*TimeWidth);%�ز��Ĳ�������=��ѹϵ������=��̬����Ŀ+1
if rem(number,2)~=0
   number=number+1;
end   
for i=-fix(number/2):fix(number/2)-1
  Chirp(i+fix(number/2)+1)=exp(j*(pi*(BandWidth/TimeWidth)*(i/Fs)^2));
  
end
coeff=conj(fliplr(Chirp));
%�ز���
SignalAll=zeros(1,TotalNumber);%����������ź�,����0
for k=1:TargetNumber% ���β�������Ŀ��
   SignalTemp=zeros(1,SampleNumber);% һ������
   SignalTemp(DelayNumber(k)+1:DelayNumber(k)+number)=sqrt(SigPower(k))*Chirp;%һ�������1��Ŀ�꣨δ�Ӷ������ٶȣ�
   Signal=zeros(1,TotalNumber);
   for i=1:PulseNumber
      Signal((i-1)*SampleNumber+1:i*SampleNumber)=SignalTemp;
   end
   FreqMove=exp(1j*2*pi*TargetFd(k)*(0:TotalNumber-1)/Fs);%Ŀ��Ķ������ٶ�*ʱ��=Ŀ��Ķ���������
   Signal=Signal.*FreqMove;
   SignalAll=SignalAll+Signal;
end

Echo = awgn(SignalAll,AWGNpower);

for i=1:PulseNumber   %�ڽ��ջ�������,���յĻز�Ϊ0
      Echo((i-1)*SampleNumber+1:(i-1)*SampleNumber+number)=0;
end
pc_time0=conv(Echo,coeff);

% Ƶ����ѹ
nfft = 2^nextpow2(length(Echo));%FFT����
Echo_fft=fft(Echo,nfft);
coeff_fft=fft(coeff,nfft);
pc_fft=Echo_fft.*coeff_fft;
pc_freq0=ifft(pc_fft);
pc_freq1=pc_freq0(number:TotalNumber+number-1);%ȥ����̬�� number-1��,����������(2048-number+1-TotalNumber=45��)

%��������š������ź���������
for i=1:PulseNumber
      pc(i,1:SampleNumber)=pc_freq1((i-1)*SampleNumber+1:i*SampleNumber);
end

%% MTI
mti = zeros(PulseNumber-1,size(pc,2));
for i=1:PulseNumber-1  %��������������һ������
   mti(i,:)=pc(i+1,:)-pc(i,:);
end
figure;
  xmti = C/2*(0:1/Fs:PRT-1/Fs);
  ymti = 1:PulseNumber-1;
mesh(xmti,ymti,abs(mti));title('MTI ���');
  xlabel('����/m');

figure;
subplot(211)
plot(xmti,abs(pc(1,:)));grid on;
title('����ѹ�����');xlabel('����/m');
subplot(212)
plot(xmti,abs(mti(1,:)));grid on;
title('MTIһ�ζ������');xlabel('����/m');

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
  mesh(xmtd,ymtd,abs(mtd));title('MTD  ���');
  xlabel('����/m');ylabel('�ٶ�/ms^-^1')
  xlim([min(xmtd) max(xmtd)]);
  ylim([min(ymtd) max(ymtd)]);
  
mtd_result = max(abs(mtd));


%% CFAR
Pfa = 1e-6;
alpha = SampleNumber*(Pfa^(-1/SampleNumber)-1);
% alpha = 1;
num = 60;
protect = 20;

cfar_result(1,1)=mean(mtd_result(1,2:(num+1)));%��1����龯����ʱ������ֵ��������num�����������
for i=2:num  %%%%��2�㵽��num��ĺ��龯������������ǰ��ͺ����num���������ͬ����
    cfar_result(1,i)=(mean(mtd_result(1,1:i-1))+mean(mtd_result(1,i+1:i+num)))/2;
end
for i=(num+1):SampleNumber-(num+1)
%���������ݵ���龯�����������ֵ����ǰ��ͺ����num��������ͬ����
    cfar_result(1,i)=max(mean(mtd_result(1,i-num:i-protect)),mean(mtd_result(1,i+protect:i+num)));
end
for i=SampleNumber-num:SampleNumber-1
%������num�㵽������2����龯�����������ֵ����ǰ��num��ͺ����������ͬ����
    cfar_result(1,i)=(mean(mtd_result(1,i-num:i-1))+mean(mtd_result(1,(i+1):SampleNumber)))/2;
end
%%%���һ��ĺ��龯�����������ֵ����ǰ���num����������� 
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
xlabel('����/m');
title('MTD�������ģ���(�ź����ͨ��)');
grid on;

subplot(212);
plot(xmtd,s_result);
xlim([min(xmtd) max(xmtd)]);
xlabel('����/m');
title('���龯������');
grid on;
 