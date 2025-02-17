function [v]=dc_compensation(z)
%Function to estimate and remove DC impairments in the IQ branch
%    v=dc_compensation(z) removes the estimated DC impairment
iDCest=mean(real(z));%estimated DC on I branch
qDCest=mean(imag(z));%estimated DC on I branch
v=z-(iDCest+1i*qDCest);%remove estimated DCs