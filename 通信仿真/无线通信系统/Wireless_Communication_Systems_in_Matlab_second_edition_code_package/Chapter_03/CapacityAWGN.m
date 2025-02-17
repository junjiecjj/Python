clearvars; clc
nBits=10^5; %number of bits to transmit
snrdB= -10:0.5:30; %Range of SNRs to simulate
C = zeros(1,length(snrdB)); %to store capacity for each point

for j=1:length(snrdB),%for each error probability
    x = rand(1,nBits)<0.5; %uniform source bits of 1's and 0's
    P=sum(abs(x).^2)/(length(x)); %Calculate actual signal power    
    
    [y,~,N0] = add_awgn_noise(x,snrdB(j));%Add noise for SNR
    %The value N0 is computed inside the add_awgn_function
    
    C(j) = log2(1+P/N0); %capacity assuming BW=1
end
plot(snrdB,C); title('Capacity over AWGN channel');
xlabel('SNR (dB)'); ylabel('Capacity (bits/channel use)');
