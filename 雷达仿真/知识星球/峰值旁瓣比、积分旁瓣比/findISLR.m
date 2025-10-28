close all;clear all;
%% LFM�źŵĲ���
T=10e-6;                          %�ź�ʱ��
B=30e6;                           %�źŴ���
K=B/T;                            %���Ե�Ƶϵ��
fc=0;                             %�ź���Ƶ
a=20;                              %����������
fs=a*B;Ts=1/fs;                   %������Fs
t0=0;                             %ʱ��
tc=0;                        %tc=0Ϊ�����źţ�tc��Ϊ0��Ϊ�ǻ����ź�
N=T/Ts;                           %��������
%% �ź�����
t=linspace(-T/2,T/2,N);
st=exp(1i*pi*K*(t-tc).^2);         %��Ƶ�ź�
ht=exp(-1i*pi*K*(t+tc).^2);        %ƥ���˲���
%% ƥ�����
sout=conv(st,ht,'same');
sout_dB=20*log10(abs(sout)/max(abs(sout)));%�����һ������ѹ��ķ��ȣ�dB��

L=length(sout_dB);
[maxdata,I]=max(sout_dB);
for i=I:I+L             %Ѱ�ҵ�һ���
    if sout_dB(i+1)>sout_dB(i)
        j=i;
        break;
    else
        continue;
    end
end
p1=sum((10.^(sout_dB(2*I-j:j)/20)).^2*(1/fs));%���깦�ʻ���
p=sum((10.^(sout_dB(round(-11/B*fs)+I:round(11/B*fs)+I)/20)).^2*(1/fs));%�ź��ܹ���
ISLR=10*log10((p-p1)/p1)