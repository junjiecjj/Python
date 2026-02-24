
function Rate = Algo2(H, Ns, P, sigma2, epsilon)
    Rate = 0;
    Nrf = Ns;
    [M, N] = size(H);
    gamma = sqrt(P / (N * Nrf));
    % Find Vrf using alg1
    F_1 = H' * H;
    Vrf = alg1(F_1, Ns, gamma^2, sigma2, epsilon);

    [N, Nrf1] = size(Vrf);
    % Find Ue and GAMMAe matrices (11)
    Heff = H * Vrf;
    Q = Vrf' * Vrf;
    % Right singular vectors,
    [U, S, V] = svd(Heff * sqrtm(pinv(Q)));
    s_values = diag(S);
    Ue = V';
    % Diagonal matrix of allocated powers to each stream
    GAMMAe = eye(Nrf) * (P/Nrf)^0.5;

    % Computing digital precoder matrix (11)
    Vd =  sqrtm(pinv(Q)) * Ue * GAMMAe;
    % Vd = (np.linalg.inv(Q)**(1/2) @ Ue @ GAMMAe).astype(np.complex128)

    % Hybrid precoder matrix (8)
    Vt = Vrf * Vd;

    % Compute analog combiner matrix of receiver (15)
    F_2 =  H * (Vt * Vt') * H';
    Wrf = alg1(F_2, Ns, 1/M, sigma2, epsilon);

    % Compute the digital combiner matrix of receiver (17)
    J = Wrf' * H * (Vt * Vt') * H' * Wrf  + sigma2 * (Wrf' * Wrf);
    Wd = pinv(J) * Wrf' * H * Vt;

    % Hybrid combiner matrix (8)
    Wt = Wrf * Wd;

    % Compute the spectral efficiency metric (4)
    Rate = log2(det(real( eye(M) + 1/sigma2 * H * Vrf * (Vd * Vd') * Vrf' * H')));

end
