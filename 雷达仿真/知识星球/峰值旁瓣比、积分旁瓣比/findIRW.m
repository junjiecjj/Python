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

maxdata=max(sout_dB);
x1=find(sout_dB>=maxdata-3);
I=length(x1);
IRW=I*(1/fs)