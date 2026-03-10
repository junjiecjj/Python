function idx = helperCommChannelDelay(x, s)
% Estimate delay using cross-correlation
    Nb = size(x, 2);
    idx = zeros(1, Nb);
    for i = 1:Nb
        [r, lags] = xcorr(x(:, i), s);
        r = r(lags >= 0);
        [~, idx(i)] = max(abs(r));
    end
end