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
%% MSK调制
% 参数设置
m = 2;          %进制数
L = 2;          %关联长度，记忆长度
h_m = 1;h_p = 2;
h = h_m/h_p;    %调制指数
sps = 32;        %每个符号样点数 sample per symbol
[g,q] = rc_pulse(sps,L);%生成升余弦脉冲函数g，及其积分函数q
sum_bit=32;
signal_bit = [ 1 0 1 1 1 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1 1 1 0 0 0 1 0 1 1 0 1];
% signal_bit = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
temp = reshape(signal_bit,log2(m),sum_bit/log2(m));%串并转换 
temp = temp';
temp = bi2de(temp,'left-msb');
symbol = (2*temp-m+1)';%码元符号,行向量        
cpm = cpm_mod(symbol,h,sps,L,q,m);
%MSK_LFM信号
s1=s0.*cpm;
%% OFDM调制
s1_ofdm=reshape(s1,2,[]);
IFFT_s1_ofdm=ifft(s1_ofdm);
tx_s1_ofdm=reshape(IFFT_s1_ofdm,1,[]);
%% SNR-OFDM_MSK_NLFM
SNR=-5:1:10;
nn=500;
for n=1:nn
    for k=1:length(SNR)
    SNR1 = SNR(k)-10*log10(sps)+10*log10(log2(m));
% SNR1 = SNR(k);
    Rx_data = awgn(tx_s1_ofdm,SNR1,'measured');
    Rx_data =Rx_data ;
    Rx_data=reshape(Rx_data,2,[]);
    Rx_complex_carrier_matrix=fft(Rx_data);
    Rx_complex_carrier_matrix=reshape(Rx_complex_carrier_matrix,1,[]);
Rx_complex_carrier_matrix = Rx_complex_carrier_matrix./s0;
%====================维特比解调======================================
correlator = phase_state(sps,m,h_m,h_p,q,L);
%形状为(state_num,m,:)表示当前状态为state_index,当前输入为input_index时的复包络
% 然后得到状态网络图，每个时刻的状态为[θn,In-1,In-2,...In-L+1]
% 当前可能的输入有m种情况，因此总的转移状态情况为[state_num,m]
[next_states,pre_states] = state_grid(L,m,h_p,h_m); 
% next_state形状为[state_num,m]，表示当前状态为state_index,当前输入为input_index时的下一个时刻的state_index
% pre_state形状为[state_num,m],表示当前状态为state_index,前一时刻状态m种可能的state_index
viterbi_symbol = viterbi_demod(Rx_complex_carrier_matrix,correlator,next_states,L,h_m,h_p,m,sps);%维特比译码符号
symbol_error_num =  length(find(viterbi_symbol ~= symbol));%误符号数
symbol_error_radio = symbol_error_num/length(symbol);%误符号率
temp = (viterbi_symbol-1+m)/2;
temp = de2bi(temp,'left-msb');
temp = temp';
viterbi_bit = reshape(temp,1,[]);%维特比译码解调所得bit
[bit_error_num(k,n),bit_error_radio(k,n)] = biterr(signal_bit,viterbi_bit);%误码数及误码率
    end
end
bit_error_radio=bit_error_radio*ones(nn,1)/nn;
figure
semilogy(SNR,bit_error_radio,'*-');
% title('维比特解调');
xlabel('SNR/dB');ylabel('误码率');