





function Cap = Alg3(H, beta, Nrf, Pt, sigma2)
    [K, N] = size(H);
    diffCap = 1;
    lastCap = -1;
    P = eye(K) * Pt/K;
    tmp = rand(N, Nrf) * 2 * pi;
    Vrf =  exp(1j * tmp);

    Ht =  sqrtm(pinv(P)) * H;
    % Ht = P^(-0.5) * H; 
    it = 0;
    epsilon = 0.001;
    while (diffCap > epsilon) & (it < 30)
        it = it + 1;
        Vrf = updateVRF(N, Nrf, Ht, Vrf);

        % abs(Vrf)
        VDt = Vrf' * H' * pinv(H * Vrf * (Vrf') * H');
        Qt = (VDt') * (Vrf') * Vrf * VDt;
        [P, ~, psum, ~] = updateP(Qt, beta, Pt, K, sigma2);
        Ht = sqrtm(pinv(P)) * H;
        % Ht = P^(-0.5) * H;
        Cap  = sum(beta .* log2(1 + diag(P).'/sigma2));
        diffCap = abs((Cap - lastCap)/Cap);
        lastCap = Cap;
        fprintf('  Alg3: it = %.6f, Cap = %.6f, diffCap = %.6f\n', it, Cap, diffCap);
    end
end














