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


%% �Ӵ�ЧӦ
M=length(ht);%���ĳ���
w=hanning(M);%�ӵĴ�����������
sout_win=conv(st,(ht.*w'),'same');%�Ӵ�������
sout_dB_win=20*log10(abs(sout_win)/max(abs(sout_win)));%�Ӵ��������һ������ѹ��ķ��ȣ�dB��

%%
figure;
subplot(221);plot(t*1e6,real(st));grid on;
title('�ź�ʵ��');xlabel('ʱ��/us');ylabel('�ź�ʵ������');
subplot(222);plot(t*1e6,imag(st));grid on;
title('�ź��鲿');xlabel('ʱ��/us');ylabel('�ź��鲿����');
subplot(223);plot(t*1e6,real(ht));grid on;
title('�˲���ʵ��');xlabel('ʱ��/us');ylabel('�˲���ʵ������');
subplot(224);plot(t*1e6,imag(ht));grid on;
title('�˲����鲿');xlabel('ʱ��/us');ylabel('�˲����鲿����');
figure;
plot(t*1e6,sout_dB_win);grid on;
title('�����ź���ѹ���');xlabel('ʱ��/us');ylabel('ѹ����ķ��ȣ�dB��');
zoom xon