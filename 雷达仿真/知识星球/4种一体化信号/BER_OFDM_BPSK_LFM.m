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
% figure(1)
% plot(t,s0);
% [afmag0,delay0,doppler0]=ambgfun(s0,fs,fs/10);
% figure(2)
% mesh(delay0,doppler0,afmag0);
% xlim([-1e-5,1e-5]);
% ylim([-4e7,4e7]);
% afmag0=10*log(afmag0);
% figure;
% plot(delay0,afmag0(1025,:),'-o');
% xlim([-0.5e-6,0.5e-6]);
% ylim([-70,0]);
% figure;
% plot(doppler0,afmag0(:,1020),'-o');
% xlim([-0.4e6,0.4e6]);
% ylim([-70,0]);

%% BPSK调制
%%基带信号产生
code = round(rand(1,1024));  % 二进制随机序列
%%BPSK基带调制
s = (code - 1/2) * 2;      % 双极性不归零序列
% s =exp(j*s*pi);
sBPSK=s.*s0;
%% OFDM调制
sBPSK_ofdm=reshape(sBPSK,2,[]);
IFFT_sBPSK_ofdm=ifft(sBPSK_ofdm);
tx_sBPSK_ofdm=reshape(IFFT_sBPSK_ofdm,1,[]);
%% 画图误码率
%% SNR-OFDM_MSK_NLFM
SNR=-5:1:10;
nn=500;
for n=1:nn
    for k=1:length(SNR)
    SNR1 = SNR(k);
% SNR1 = SNR(k);
    Rx_data = awgn(tx_sBPSK_ofdm,SNR1,'measured');
    Rx_data =Rx_data ;
    Rx_data=reshape(Rx_data,2,[]);
    Rx_complex_carrier_matrix=fft(Rx_data);
    Rx_complex_carrier_matrix=reshape(Rx_complex_carrier_matrix,1,[]);
Rx_complex_carrier_matrix = Rx_complex_carrier_matrix./s0;
%====================维特比解调======================================
rx_dem = 2*floor(real(Rx_complex_carrier_matrix))/ + 1/2;
data_dem = (rx_dem+1)/2;

data_dem(data_dem>0)=1;
data_dem(data_dem<0)=0;
% viterbi_bit=viterbi_bit/2+1/2;

[bit_error_num(k,n),bit_error_radio(k,n)] = biterr(code,data_dem);%误码数及误码率
    end
end
bit_error_radio=bit_error_radio*ones(nn,1)/nn;
figure
semilogy(SNR,bit_error_radio,'*-');
% title('维比特解调');
xlabel('SNR/dB');ylabel('误码率');






















