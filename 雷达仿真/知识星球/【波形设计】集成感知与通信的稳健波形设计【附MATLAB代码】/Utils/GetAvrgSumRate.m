%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function SumRate = GetAvrgSumRate(H, X, S, N0)
% Calculate Average Achievable Sum-Rate For Downlink Communications
    [K, L] = size(S);
    gamma = zeros(K, 1);
    for i = 1:K
        signal_power        = 0;
        interference_power  = 0;
        for j = 1:L
            signal_power        = signal_power + (abs(S(i, j)))^2;
            interference_power  = interference_power + (abs(H(i,:) * X(:, j) - S(i, j)))^2;
        end
        signal_power        = signal_power/L;
        interference_power  = interference_power/L;
        noise_power         = N0;

        gamma(i) = signal_power/(interference_power + noise_power);
    end

    SumRate = mean(log2(1 + gamma));
end

