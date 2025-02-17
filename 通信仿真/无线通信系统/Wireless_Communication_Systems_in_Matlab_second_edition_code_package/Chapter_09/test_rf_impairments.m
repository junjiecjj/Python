clearvars; clc;
M=64;%M-QAM modulation order
N=1000;%To generate random symbols
d=ceil(M.*rand(N,1));%random data symbol generation
s = mqam_modulator(M,d); %M-QAM modulated symbols (s)
 
g_1=0.8; phi_1=0; dc_i_1=0;   dc_q_1=0; %gain mismatch only
g_2=1;   phi_2=12; dc_i_2=0;   dc_q_2=0; %phase mismatch only
g_3=1;   phi_3=0; dc_i_3=0.5; dc_q_3=0.5; %DC offsets only
g_4=0.8; phi_4=12; dc_i_4=0.5; dc_q_4=0.5; %All impairments

r1=receiver_impairments(s,g_1,phi_1,dc_i_1,dc_q_1);
r2=receiver_impairments(s,g_2,phi_2,dc_i_2,dc_q_2);
r3=receiver_impairments(s,g_3,phi_3,dc_i_3,dc_q_3);
r4=receiver_impairments(s,g_4,phi_4,dc_i_4,dc_q_4);

subplot(2,2,1);plot(real(s),imag(s),'b.');hold on; 
plot(real(r1),imag(r1),'r.'); title('IQ Gain mismatch only')
subplot(2,2,2);plot(real(s),imag(s),'b.');hold on; 
plot(real(r2),imag(r2),'r.'); title('IQ Phase mismatch only')
subplot(2,2,3);plot(real(s),imag(s),'b.');hold on; 
plot(real(r3),imag(r3),'r.'); title('DC offsets only')
subplot(2,2,4);plot(real(s),imag(s),'b.');hold on; 
plot(real(r4),imag(r4),'r.');title('IQ impairments & DC offsets');