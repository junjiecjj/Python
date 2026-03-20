%% Waterfilling Power allocation for Gaussian signals
% ---------------------Model---------------------
% y_i = h_i * sqrt(p_i) * s_i + w_i, {i <= N}
% w_i is an AWGN
% s_i is IID Gaussian, E{s_i} = 0
% ---------------------Input---------------------
% h: a real vector, h = [h_1, ..., h_N]
% snr: a scalar / vector with the same size as h (not in dB)
% pow: a positive scalar, sum(p_i) = pow
% --------------------Output---------------------
% p_wf: the WF power allocation vector
% -----------------------------------------------
function p_wf = WF_Gauss(snr, h, pow)
    N = length(h);
    % Sort gam in descending order
    gam = snr .* abs(h).^2;
    [gam, s_ind] = sort(gam, 'descend');
    % Binary search 
    low = 1;
    high = N;
    while low <= high
        K = floor((low + high)/2);
        tmp = (pow + sum(1./gam(1:K))) / K;
        eta = 1 / tmp;                      % 1/eta is water level
        if K == N && eta < gam(N)
            break
        elseif eta < gam(K) && eta >= gam(K+1)
            break
        elseif eta >= gam(K)
            high = K - 1;
        else
            low = K + 1;
        end
    end
    % WF
    % min_power = 1e-3;  % 设置最小功率阈值
    % p_wf = max(1 / eta - 1 ./ gam, min_power);
    p_wf = 1 / eta - 1 ./ gam;
    p_wf(p_wf<=0) = 0;
    p_wf(s_ind) = p_wf;
end