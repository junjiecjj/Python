% create fading across OFDM subcarriers

%pg 28
clc; clear;
tau = [0 200 800 1200 2300 3700]*1e-9;
p_db = [0 -0.9 -4.9 -8.0 -7.8 -23.9];
p = 10.^(p_db/10);
path_num = length(p);

p = p/sum(p);
ampli = sqrt(p);

BW = 10e6;
subc_num = 8;
df = BW/subc_num;
% subc_num = 600;
% df = 15000;
fc = (1:subc_num)*df;

m = 1/sqrt(2)*( randn(1,path_num) + 1i*randn(1,path_num) );
m = m.*ampli;
phase = exp(-1i*2*pi*tau'*fc);
fading_en = sum(diag(m)*phase);
fading_p = fading_en.*conj(fading_en);
fading_p_db = 10*log10(fading_p);
plot(fading_p_db);
grid on;
xlabel('subcarrier'); ylabel('multipath fading (dB)');
