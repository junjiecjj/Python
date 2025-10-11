%% ��������
clear all;clc ;close all 

fs=100e6;%100Mhz
fc=10e6;%10Mhz
T=10.24e-6;
B=40e6;
N=T*fs;
t=-T/2:1/fs:T/2-1/fs;
f=(-N/2:N/2-1)*B/N;
%% LFM�ź�
K=B/T;
s0=exp(j*2*pi*fc+j*pi*K*t.^2);
%% MSK����
% ��������
m = 2;          %������
L = 2;          %�������ȣ����䳤��
h_m = 1;h_p = 2;
h = h_m/h_p;    %����ָ��
sps = 32;        %ÿ������������ sample per symbol
[g,q] = rc_pulse(sps,L);%�������������庯��g��������ֺ���q
sum_bit=32;
signal_bit = [ 1 0 1 1 1 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1 1 1 0 0 0 1 0 1 1 0 1];
% signal_bit = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
temp = reshape(signal_bit,log2(m),sum_bit/log2(m));%����ת�� 
temp = temp';
temp = bi2de(temp,'left-msb');
symbol = (2*temp-m+1)';%��Ԫ����,������        
cpm = cpm_mod(symbol,h,sps,L,q,m);
%MSK_LFM�ź�
s1=s0.*cpm;
%% OFDM����
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
%====================ά�رȽ��======================================
correlator = phase_state(sps,m,h_m,h_p,q,L);
%��״Ϊ(state_num,m,:)��ʾ��ǰ״̬Ϊstate_index,��ǰ����Ϊinput_indexʱ�ĸ�����
% Ȼ��õ�״̬����ͼ��ÿ��ʱ�̵�״̬Ϊ[��n,In-1,In-2,...In-L+1]
% ��ǰ���ܵ�������m�����������ܵ�ת��״̬���Ϊ[state_num,m]
[next_states,pre_states] = state_grid(L,m,h_p,h_m); 
% next_state��״Ϊ[state_num,m]����ʾ��ǰ״̬Ϊstate_index,��ǰ����Ϊinput_indexʱ����һ��ʱ�̵�state_index
% pre_state��״Ϊ[state_num,m],��ʾ��ǰ״̬Ϊstate_index,ǰһʱ��״̬m�ֿ��ܵ�state_index
viterbi_symbol = viterbi_demod(Rx_complex_carrier_matrix,correlator,next_states,L,h_m,h_p,m,sps);%ά�ر��������
symbol_error_num =  length(find(viterbi_symbol ~= symbol));%�������
symbol_error_radio = symbol_error_num/length(symbol);%�������
temp = (viterbi_symbol-1+m)/2;
temp = de2bi(temp,'left-msb');
temp = temp';
viterbi_bit = reshape(temp,1,[]);%ά�ر�����������bit
[bit_error_num(k,n),bit_error_radio(k,n)] = biterr(signal_bit,viterbi_bit);%��������������
    end
end
bit_error_radio=bit_error_radio*ones(nn,1)/nn;
figure
semilogy(SNR,bit_error_radio,'*-');
% title('ά���ؽ��');
xlabel('SNR/dB');ylabel('������');