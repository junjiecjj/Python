clear;
clc;
close all;
%--------------Main--------------%
c_3d = zeros(10,10);
%--------------接收天线变化--------------%
N = 4;%发射天线
cap = zeros(1,20);
%SNR为5dB
SNR = 5; %SNR in dB
for i = 1:1:20
    cap(1,i) = mimo_capacity(N,i,SNR);
end
figure(1);
plot(1:1:20,cap,'b')
title('发射天线数目为4的情况')
xlabel('接收天线数目')
ylabel('信道容量 bits/s/Hz')
hold on

%SNR为10dB
SNR = 10;
for i = 1:1:20
    cap(1,i) = mimo_capacity(N,i,SNR);
end
%figure
plot(1:1:20,cap,'r')
hold on

%SNR为15dB
SNR = 15;
for i = 1:1:20
    cap(1,i) = mimo_capacity(N,i,SNR);
end
%figure
plot(1:1:20,cap,'g')
hold on

SNR = 20;
for i = 1:1:20
    cap(1,i) = mimo_capacity(N,i,SNR);
end
%figure
plot(1:1:20,cap,'m')
legend('5db','10dB','15dB','20db')

%--------------发射天线变化--------------%
M = 4;%发射天线
cap = zeros(1,20);
%SNR为5dB
SNR = 5; %SNR in dB
for i = 1:1:20
    cap(1,i) = mimo_capacity(i,M,SNR);
end
figure(2);
plot(1:1:20,cap,'b')
title('接收天线数目为4的情况')
xlabel('发射天线数目')
ylabel('信道容量 bits/s/Hz')
hold on

%SNR为10dB
SNR = 10;
for i = 1:1:20
    cap(1,i) = mimo_capacity(i,M,SNR);
end
%figure
plot(1:1:20,cap,'r')
hold on

%SNR为15dB
SNR = 15;
for i = 1:1:20
    cap(1,i) = mimo_capacity(i,M,SNR);
end
plot(1:1:20,cap,'g')
hold on

SNR = 20;
for i = 1:1:20
    cap(1,i) = mimo_capacity(i,M,SNR);
end
%figure
plot(1:1:20,cap,'m')
legend('5db','10dB','15dB','20db')

%--------------三维图展示MIMO----------------%
SNR = 15;
for i = 1:1:10
    for j = 1:1:10
        c_3d(i,j) = mimo_capacity(i,j,SNR);
    end
end
figure
mesh(1:1:10,1:1:10,c_3d)
title('3-dimension visual')

%---------------AWGN SISO信道--------------%
cap_awgn = zeros(1,31);
for SNR = -10:1:20
    cap_awgn(1,SNR+11) = awgn_capacity(SNR);
end
figure
plot(1:1:31,cap_awgn)
title('awgn信道容量')
xlabel('SNR in dB')
ylabel('信道容量 bits/s/Hz')

%---------------Raly SISO信道--------------%
c = zeros(1,31);

for SNR = 0:1:30
    c(1,SNR+1) = ralychannel(SNR);
end
figure
plot(1:1:31,c)
title('瑞利信道容量')
xlabel('SNR in dB')
ylabel('信道容量 bits/s/Hz')

%------------------Alamouti---------------%
c = zeros(1,31);

for SNR = 0:1:30
    c(1,SNR+1) = alamouti(SNR);
end
figure
plot(1:1:31,c)
title('Alamouti码 2发1收')
xlabel('SNR in dB')
ylabel('信道容量 bits/s/Hz')




