function rate = average_user_rate(H, X, S, N0)
    E = H * X - S;
    K = size(S, 1);
    gamma = zeros(K, 1);
    for idxK = 1:K
        signalPower = mean(abs(S(idxK, :)).^2);
        muiPower = mean(abs(E(idxK, :)).^2);
        gamma(idxK) = signalPower / (muiPower + N0);
    end
    rate = mean(log2(1 + gamma));
end