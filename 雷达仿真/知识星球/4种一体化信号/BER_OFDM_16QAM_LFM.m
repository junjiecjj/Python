%% 参数设置
clear all;clc ;close all 

fs=100e6;%100Mhz
fc=10e6;%10Mhz
T=10.24e-6;
B=40e6;
N=T*fs;
t=-T/2:1/fs:T/2-1/fs;
f=(-N/2:N/2-1)*B/N;
%% LFM信号
K=B/T;
s0=exp(j*2*pi*fc+j*pi*K*t.^2);

%% 16QAM调制
nsymbol=1024;%表示一共有多少个符号，这里定义100000个符号
M=16;%M表示QAM调制的阶数,表示16QAM，16QAM采用格雷映射(所有星座点图均采用格雷映射)
graycode=[0 1 3 2 4 5 7 6 12 13 15 14 8 9 11 10];%格雷映射编码规则
msg=randi([0,M-1],1,nsymbol);%0到15之间随机产生一个数,数的个数为：1乘nsymbol，得到原始数据
msg1=graycode(msg+1);%对数据进行格雷映射
msgmod=qammod(msg1,M);%调用matlab中的qammod函数，16QAM调制方式的调用(输入0到15的数，M表示QAM调制的阶数)得到调制后符号
sQAM=msgmod;
sQAM=sQAM.*s0;
%% OFDM调制
sQAM_ofdm=reshape(sQAM,2,[]);
IFFT_sQAM_ofdm=ifft(sQAM_ofdm);
tx_QAM_ofdm=reshape(IFFT_sQAM_ofdm,1,[]);

%% 画图误码率
%% SNR-OFDM_MSK_NLFM
SNR=-5:1:10;
nn=500;
for n=1:nn
    for k=1:length(SNR)
    SNR1 = SNR(k);
% SNR1 = SNR(k);
    Rx_data = awgn(tx_QAM_ofdm,SNR1,'measured');
    Rx_data =Rx_data ;
    Rx_data=reshape(Rx_data,2,[]);
    Rx_complex_carrier_matrix=fft(Rx_data);
    Rx_complex_carrier_matrix=reshape(Rx_complex_carrier_matrix,1,[]);
Rx_complex_carrier_matrix = Rx_complex_carrier_matrix./s0;
%====================维特比解调======================================
y=qamdemod(Rx_complex_carrier_matrix,M);
decmsg=graycode(y+1);
[bit_error_num(k,n),bit_error_radio(k,n)] = biterr(msg,decmsg);%误码数及误码率
    end
end
bit_error_radio=bit_error_radio*ones(nn,1)/nn;
figure
semilogy(SNR,bit_error_radio,'*-');
% title('维比特解调');
xlabel('SNR/dB');ylabel('误码率');
