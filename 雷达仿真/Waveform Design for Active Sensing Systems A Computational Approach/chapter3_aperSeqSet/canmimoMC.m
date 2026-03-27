function [X XInit] = canmimoMC(N, M)
% [X XInit] = canmimoMC(N, M), find the Multi-CAN sequence set with lowest peak
% sidelobe level from 100 Monte-Carlo trials
%   N: length of each transmit sequence
%   M: number of transmit sequences
%   X: N-by-M, the obained Multi-CAN sequence set
%   XInit: N-by-M, the corresponding initialization sequence set

XInit = zeros(N,M);
peakSidelobe = N;

for k = 1:100
    X0 = exp(1i * 2*pi * rand(N,M));
    Y = canmimo(N, M, X0);
    [peakAuto peakCross] = mimocriterion(Y);
    peak = max([peakAuto peakCross]);
    if peak < peakSidelobe
        peakSidelobe = peak;
        X = Y;
        XInit = X0;
    end
end