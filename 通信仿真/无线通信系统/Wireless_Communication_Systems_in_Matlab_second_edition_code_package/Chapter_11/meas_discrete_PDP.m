function [meanDelay,rmsDelay,symbolRate,coherenceBW] = meas_discrete_PDP(Ps,TAUs)
%Calculate mean Delay, RMS delay spread and the maximum symbol rate
%that a signal can be transmitted without ISI and the coherence BW
%for the discrete PDP. The discrete PDP is specified as a list of
%power values (Ps) in dBm and corresponding time delays (TAUs)
p_i = 10.^(Ps/10);%convert dBm to linear values
meanDelay = sum(p_i.*TAUs)/sum(p_i);
rmsDelay = sqrt(sum(p_i.*(TAUs-meanDelay).^2)/sum(p_i));
symbolRate = 1/(10*rmsDelay);%Recommended max sym rate to avoid ISI
coherenceBW = 1/(50*rmsDelay);%0.9 correlation
end