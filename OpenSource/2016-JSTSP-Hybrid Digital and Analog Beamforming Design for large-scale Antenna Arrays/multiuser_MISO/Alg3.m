





function Cap = Alg3(H, beta, Nrf, Pt, sigma2)
    [K, N] = size(H);
    diffCap = 1;
    lastCap = -1;
    P = eye(K) * Pt/K;
    tmp = rand(N, Nrf) * 2 * pi;
    Vrf =  exp(1j * tmp);

    Htilde =  sqrtm(pinv(P)) * H;
    % Htilde = P^(-0.5) * H; 
    it = 0;
    epsilon = 0.001;
    while (diffCap > epsilon) & (it < 30)
        it = it + 1;
        Vrf = updateVRF(N, Nrf, Htilde, Vrf);

        % abs(Vrf)
        VD_tilde = Vrf' * H' * pinv(H * Vrf * (Vrf') * H');
        Q_tilde = (VD_tilde') * (Vrf') * Vrf * VD_tilde;
        [P, ~] = updateP(Q_tilde, beta, Pt, K, sigma2);
        
        Htilde = sqrtm(pinv(P)) * H;
        % Htilde = P^(-0.5) * H;
        Cap  = sum(beta .* log2(1 + diag(P)/sigma2));
        diffCap = abs((Cap - lastCap)/Cap);
        lastCap = Cap;
    end
end














