%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function Pd = DetectProb(a, X, N0, Pfa)
% Calculate the "Detection Probability" of a target using the given waveform 
%        Pd : probability of detection
%         a : steering vector
%         X : waveform
%       Pfa : probability of false alarm
%
% see: Khawar, A., Abdelhadi, A., & Clancy, C. (2015). Target detection performance of spectrum sharing MIMO radars. IEEE Sensors Journal, 15(9), 4928-4940.
%      - This function implements Eq. (69) inside
%
    [~, L] = size(X);
    R = X*X'/L;
    delta = (0.9)^2*(abs(a'*R*a))^2/N0;                 % non-centrality parameter of "non-central chi-square cumulative distribution function (cdf)"
    Pd = 1 - ncx2cdf(chi2inv(1 - Pfa, 2), 2, delta);
%
% Note: I am not sure why there is a "square" over "abs(a'*R*a)". Am I wrong? Please confirm this.
end

