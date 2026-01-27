function H = channel(K, N, M, L)
    H = zeros(K, M, N);
    for k = 1:K
        phi_t = 2*pi*rand(L);
        phi_r = 2*pi*rand(L);
        alphas = randn(L);
        Hk = zeros(M, N);
        for l = 1:L 
            at = stevec_ULA(phi_t(l), N);
            ar = stevec_ULA(phi_r(l), M);
            Hk = Hk + alphas(l) * (ar * at');
        end
        H(k,:,:) = Hk;
    end
    H = H * sqrt(N*M/L); 
end