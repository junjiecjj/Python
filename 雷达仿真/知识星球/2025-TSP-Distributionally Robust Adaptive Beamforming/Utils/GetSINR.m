%{
    Online supplementary materials of the paper titled:
    Distributionally Robust Adaptive Beamforming
    By Shixiong Wang, Wei Dai, and Geoffrey Ye Li
    From Department of Electrical and Electronic Engineering, Imperial College London

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     8 Oct 2024, 13 March 2025
    @Home:     https://github.com/Spratm-Asleaf/Beamforming-UDL
%}

function sinr = GetSINR(w, a0, Ripn0, Ps)
% Calculate the real-world SINR of a given beamformer w
% SINR(w) = (Ps*w'*(a0*a0')*w)/(w'*Ripn0*w)
%   w:     beamformer of interest
%   a0:    true steer vector
%   Ripn0: true covariance of IPN
%   Ps:    power of signal of interest

    % For a given beamformer w, its real-world performance is evaluated under ground truth a0 and Ripn0
    sinr = abs((Ps*w'*(a0*a0')*w)/(w'*Ripn0*w));

    % In Decibel
    sinr = 10*log10(sinr);
end

