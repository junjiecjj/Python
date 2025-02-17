%Eb/N0 Vs SER for BPSK over Rayleigh flat-fading with transmit diversity
clearvars;clc;
%---------Input Fields------------------------
nSym = 10e5; %Number of symbols to transmit
EbN0dBs = -20:2:36; %Eb/N0 range in dB for simulation 
M = 2; %M-ary PSK modulation
MOD_TYPE = 'PSK'; %Modulation type

k=log2(M);EsN0dBs = 10*log10(k)+EbN0dBs; %EsN0dB calculation
ser_sim = zeros(1,numel(EsN0dBs));q=1; %simulated symbol error rates

for EsN0dB = EsN0dBs %simulate for each # of received branches    
    %---------- Transmitter -----------
    d = ceil(M*rand(nSym,1));%uniform random symbols 1:M
    s = modulate(MOD_TYPE,M,d);%Modulation 
    ss = kron(reshape(s,2,[]),ones(1,2));%shape as 2xnSym vector
    
    h = sqrt(1/2)*(randn(2,nSym/2)+1j*randn(2,nSym/2));%channel coeffs
    H = kron(h,ones(1,2)); %shape as 2xnSym vector
    H(:,2:2:end) = conj(flipud(H(:,2:2:end)));
    H(2,2:2:end) = -1*H(2,2:2:end);%Alamouti coded channel coeffs
    
    signal = sum(H.*ss,1);%effect of channel on the modulated signal
    
    gamma = 10.^(EsN0dB/10);%for AWGN noise addition
    P = sum(abs(signal).^2,2)./nSym; %calculate power in each branch of signal
    N0 = P/gamma; %required noise spectral density for each branch
    %Scale each row of noise with the calculated noise spectral density
    noise = (randn(size(signal))+1j*randn(size(signal))).*sqrt(N0/2);
    
    r = signal+noise; %received signal
    
    %Receiver processing
    rVec = kron(reshape(r,2,[]),ones(1,2)); %2xnSym format
    
    Hest = H; %perfect channel estimation
    Hest(1,1:2:end) = conj(Hest(1,1:2:end));
    Hest(2,2:2:end) = conj(Hest(2,2:2:end));%Hermitian transposed Hest matrix
    
    y = sum(Hest.*rVec,1); %combiner output
    sCap = y./sum(conj(Hest).*Hest,1);%decision vector for demod
    dCap = demodulate(MOD_TYPE,M,sCap).';%demodulation
    ser_sim(q) = sum(dCap ~= d)/nSym;q=q+1;%error rate calculation    
end
figure;
semilogy(EbN0dBs,ser_sim,'r','lineWidth',1.5);hold on;%plot simulated error rates
title('2x1 Transmit diversity - Alamouti coding');grid on;
xlabel('EbN0 (dB)');ylabel('Symbol error rate (Ps)');   