%DCMC capacity for M-ary modulations on AWGN or Rayleigh channel
clearvars; clc;
%---------Input Fields------------------------
nSym=10^4;%Number of symbols to transmit
channelModel = 'AWGN'; %channel model - 'AWGN' or 'RAYLEIGH'
snrdB = -10:1:30; % SNRs in dB for noise generation
MOD_TYPE='PAM'; %Set 'PSK' or 'QAM' or 'PAM'
arrayOfM=[2,4,8,16]; %array of M values to simulate
%arrayOfM=[4,16,64,256]; %array of M values for MOD_TYPE='QAM'

plotColor =['b','g','c','m','k']; j=1; %plot colors/color index
legendString = cell(1,length(arrayOfM)); %for legend entries
for M = arrayOfM
    C = zeros(1,length(snrdB));%capacity

    d = ceil(M.*rand(1,nSym));%uniformly distributed source syms
    [s,constellation]=modulate(MOD_TYPE,M,d);%constellation mapping
    for i=1:length(snrdB),
        if strcmpi(channelModel,'RAYLEIGH'),%rayleigh flat channel
            h = 1/sqrt(2)*(randn(1,nSym)+1i*randn(1,nSym));
        else %else assume no channel effect
            h = ones(1,nSym);
        end
        hs = h.*s; %channel effect on the modulated symbols
        [r,~,N0] = add_awgn_noise(hs,snrdB(i));%r = h*s+n (received)
        %Calculate conditional probabilities of each const. point
        pdfs = exp(-(abs(ones(M,1)*r - constellation.'*h).^2)/N0);
        p = max(pdfs,realmin);%prob of each constellation points
        p = p./ (ones(M,1)*sum(p)); %normalize probabilities
        symEntropy = -sum(p.*log2(p)); %senders uncertainity
        C(i)=log2(M)-mean(symEntropy);%bits/sym-senders uncertainity
    end
    plot(snrdB,C,'LineWidth',1.0,'Color',plotColor(j)); hold on;
    legendString{j}=[num2str(M),'-', MOD_TYPE];j=j+1;
end
legend(legendString);
title(['Constrained Capacity for ', MOD_TYPE,' on ', channelModel, ' channel']);
xlabel('SNR (dB)'); ylabel('Capacity (bits/sym)');
