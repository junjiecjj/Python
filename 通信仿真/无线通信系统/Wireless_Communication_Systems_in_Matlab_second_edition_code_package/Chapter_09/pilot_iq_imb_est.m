function [Kest,Pest]=pilot_iq_imb_est(g,phi,dc_i,dc_q)
%Length 64 - Long Preamble as defined in the IEEE 802.11a
preamble_freqDomain = [0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,...
    -1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,...
    0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,...
    -1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0];%freq. domain representation
preamble=ifft(preamble_freqDomain,64);%time domain representation

%send known preamble through DC & IQ imbalance model and estimate it
r=receiver_impairments(preamble,g,phi,dc_i,dc_q);
z=dc_compensation(r); %remove DC imb. before IQ imbalance estimation
%IQ imbalance estimation
I=real(z); Q=imag(z);
Kest = sqrt(sum((Q.*Q))./sum(I.*I)); %estimate gain imbalance
Pest = sum(I.*Q)./sum(I.*I); %estimate phase mismatch
end
