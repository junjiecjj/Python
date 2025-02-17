%Eb/N0 Vs SER for BPSK over Rayleigh flat-fading with receive diversity
clearvars;clc;
%---------Input Fields------------------------
nSym = 10e5; %Number of symbols to transmit
N = [1,2,20]; %number of diversity branches
EbN0dBs = -20:2:36; %Eb/N0 range in dB for simulation 
M = 2; %M-ary PSK modulation
MOD_TYPE = 'PSK'; %Modulation type

k=log2(M);EsN0dBs = 10*log10(k)+EbN0dBs; %EsN0dB calculation
figure;
for nRx = N %simulate for each # of received branchs
    ser_MRC = zeros(1,numel(EsN0dBs));%simulated symbol error rates
    ser_EGC = zeros(1,numel(EsN0dBs));
    ser_SC = zeros(1,numel(EsN0dBs));q=1;    
    %---------- Transmitter -----------
    d = ceil(M*rand(1,nSym));%uniform random symbols 1:M
    s = modulate(MOD_TYPE,M,d);%Modulation   
    s_diversity = kron(ones(nRx,1),s);%nRx signal branches    
     
    for EsN0dB = EsN0dBs
         h = sqrt(1/2)*(randn(nRx,nSym)+1j*randn(nRx,nSym));%Rayleigh flat-fading
         signal = h.*s_diversity; %effect of channel on the modulated signal
 
         gamma = 10.^(EsN0dB/10);%for AWGN noise addition
         P = sum(abs(signal).^2,2)./nSym; %calculate power in each branch of signal
         N0 = P/gamma; %required noise spectral density for each branch
         %Scale each row of noise with the calculated noise spectral density
         noise = (randn(size(signal))+1j*randn(size(signal))).*sqrt(N0/2);
         
         r = signal+noise;%received signal branches
         
         %MRC processing assuming perfect channel estimates
         s_MRC = sum(conj(h).*r,1)./sum(abs(h).^2,1); %detector decisions        
         d_MRC = demodulate(MOD_TYPE,M,s_MRC); %demodulation decisions             
         
         %EGC processing assuming perfect channel estimates
         h_phases = exp(-1j*angle(h));%estimated channel phases         
         s_EGC = sum(h_phases.*r,1)./sum(abs(h),1); %detector decisions        
         d_EGC = demodulate(MOD_TYPE,M,s_EGC); %demodulation decisions         
         
         %SC processing assuming perfect channel estimates
         [h_max,idx] = max(abs(h),[],1); %max |h| values along all branches         
         h_best = h(sub2ind(size(h),idx,1:size(h,2)));%best path's h estimate
         y = r(sub2ind(size(r), idx, 1:size(r,2)));%selected output
         
         s_SC = y.*conj(h_best)./abs(h_best).^2; %detector decisions         
         d_SC = demodulate(MOD_TYPE,M,s_SC); %demodulation decisions
         
         ser_MRC(q) = sum(d_MRC ~= d)/nSym;%Error rates computation
         ser_EGC(q) = sum(d_EGC ~= d)/nSym;
         ser_SC(q) = sum(d_SC ~= d)/nSym;q=q+1;         
    end    
    semilogy(EbN0dBs,ser_MRC,'b-','lineWidth',1.5);hold on;
    semilogy(EbN0dBs,ser_EGC,'r--','lineWidth',1.5);
    semilogy(EbN0dBs,ser_SC,'g-.','lineWidth',1.5);
end        
xlim([-20,36]);ylim([0.00001,1.1]);
xlabel('Eb/N0(dB)');ylabel('Symbol Error Rate (P_s)')
title('Receive diversity schemes in Rayleigh flat-fading')