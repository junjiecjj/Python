%{
    Online supplementary materials of the paper titled:
    Distributionally Robust Adaptive Beamforming
    By Shixiong Wang, Wei Dai, and Geoffrey Ye Li
    From Department of Electrical and Electronic Engineering, Imperial College London

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     8 Oct 2024, 13 March 2025
    @Home:     https://github.com/Spratm-Asleaf/Beamforming-UDL
%}

function Ripn = GetEstimatedRipn(Rx, ThetaIPN, a)
% Get Estimated Covariance Matrix of Interference-Plus-Noise
    len = length(ThetaIPN);

    DeltaTheta = ThetaIPN(2) - ThetaIPN(1);

    [N, ~] = size(Rx);
    Ripn = zeros(N, N);

    for i = 1:len
        % NB: No worry about any constants, it will be cancelled in the normalization phase
        Ripn = Ripn + (a(N, ThetaIPN(i))*a(N, ThetaIPN(i))')/(abs(a(N, ThetaIPN(i))'* (Rx)^-1 *a(N, ThetaIPN(i))));
    end

    Ripn = Ripn * DeltaTheta;
end