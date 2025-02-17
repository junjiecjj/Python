SNR = 0:1:100;%vary SNR
eta = log2(1+SNR);%find spectral efficiency
plot(SNR,eta);
xlabel('SNR');ylabel('Spectral efficiency - \eta');