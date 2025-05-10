clear;
clc;
close all;
snr = -2:10;
R = 1/2;
sigma = 1/sqrt(2 * R) * 10.^(-snr/20);
n = 8;
N = 2^n;
for i=1:length(snr)
    C_AWGN(i) = get_AWGN_capacity(1, sigma(i));
end
plot(snr,C_AWGN);
xlabel('{\it E_b/N}_0(dB)');%横坐标标号
ylabel('Capacity');%纵坐标标号







